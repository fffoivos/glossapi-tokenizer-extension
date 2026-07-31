#!/usr/bin/env python3
"""Materialize the pinned HF v2 snapshot while reusing identical donor shards.

The July-15 Clariden release and the bibliography-cleaned public revision share
422 of 431 data shards. Matching payloads are hard-linked after checksum
verification; only changed or absent files are downloaded and copied into a
symlink-free output tree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path


REPO_ID = "fffoivos/glossapi-greek-nanochat-pretraining-dataset-v2"
REVISION = "3f97cec48af502f4996cf8ff20b02660e2dd3d31"
MANIFEST_SHA = "2368c479a31341d042e8dfca8ab991dfe02bff85efbc65f8c5db29ca2418e659"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--donor-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN is required for the manually gated dataset")
    if args.output_root.exists():
        raise FileExistsError(f"refusing to replace output: {args.output_root}")
    from huggingface_hub import hf_hub_download

    manifest_cached = Path(
        hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            revision=REVISION,
            filename="manifests/deduplicated_manifest.json",
            cache_dir=args.cache_dir,
            token=os.environ["HF_TOKEN"],
        )
    )
    if sha256_file(manifest_cached) != MANIFEST_SHA:
        raise ValueError("published manifest bytes drift")
    manifest = json.loads(manifest_cached.read_text(encoding="utf-8"))
    files = manifest.get("files", [])
    if len(files) != 431 or int(manifest.get("rows", -1)) != 51839746:
        raise ValueError("published manifest accounting drift")

    temporary = Path(
        tempfile.mkdtemp(prefix=f".{args.output_root.name}.", dir=args.output_root.parent)
    )
    linked = downloaded = 0
    try:
        for row in sorted(files, key=lambda item: item["path"]):
            relative = Path(row["path"])
            destination = temporary / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            donor = args.donor_root / relative
            reusable = (
                donor.is_file()
                and not donor.is_symlink()
                and donor.stat().st_size == int(row["bytes"])
                and sha256_file(donor) == row["sha256"]
            )
            if reusable:
                os.link(donor, destination)
                linked += 1
            else:
                cached = Path(
                    hf_hub_download(
                        repo_id=REPO_ID,
                        repo_type="dataset",
                        revision=REVISION,
                        filename=relative.as_posix(),
                        cache_dir=args.cache_dir,
                        token=os.environ["HF_TOKEN"],
                    )
                )
                shutil.copyfile(cached, destination)
                shutil.copystat(cached, destination)
                downloaded += 1
            if destination.stat().st_size != int(row["bytes"]) or sha256_file(destination) != row["sha256"]:
                raise ValueError(f"materialized shard drift: {relative}")

        for filename in (
            "README.md",
            "manifests/bibliography_reconstruction.json",
            "manifests/deduplicated_manifest.json",
            "manifests/license_override_receipt.json",
            "manifests/token_counts.json",
        ):
            cached = Path(
                hf_hub_download(
                    repo_id=REPO_ID,
                    repo_type="dataset",
                    revision=REVISION,
                    filename=filename,
                    cache_dir=args.cache_dir,
                    token=os.environ["HF_TOKEN"],
                )
            )
            destination = temporary / filename
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(cached, destination)
        if sha256_file(temporary / "manifests/deduplicated_manifest.json") != MANIFEST_SHA:
            raise ValueError("materialized manifest drift")
        receipt = {
            "schema_version": "greek_cpt_hf_v2_materialization_v1",
            "status": "completed",
            "repo_id": REPO_ID,
            "revision": REVISION,
            "published_manifest_sha256": MANIFEST_SHA,
            "files": len(files),
            "rows": sum(int(row["rows"]) for row in files),
            "bytes": sum(int(row["bytes"]) for row in files),
            "hardlinked_from_donor": linked,
            "downloaded": downloaded,
            "donor_root": str(args.donor_root.resolve()),
        }
        (temporary / "materialization_receipt.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, args.output_root)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps({"ok": True, "output": str(args.output_root), "linked": linked, "downloaded": downloaded}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
