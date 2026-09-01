#!/usr/bin/env python3
"""Snapshot and cluster a reserve from the Greek Parliament Proceedings corpus.

The source CSV stores one speech fragment per row.  Validation partitions must
not split fragments from the same parliamentary sitting, so this producer uses
the sitting as the document-cluster identity and selects complete clusters.
The reserve is intentionally larger than the final 10--20M-token panel; the
cross-dedup finalizer chooses the final clusters only after exact and MinHash
comparison with the frozen training corpus.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import io
import json
import os
import re
import sys
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path
from typing import BinaryIO, Iterable


DEFAULT_RECORD_ID = 2_587_904
DEFAULT_FILE_NAME = "Greek_Parliament_Proceedings_1989_2019.csv.zip"
DEFAULT_FILE_BYTES = 471_233_955
DEFAULT_FILE_MD5 = "461dc78146a0386a6a99b4f647710a37"
DEFAULT_RESERVE_CLUSTERS = 512
SOURCE_ID = "zenodo_greek_parliament_proceedings_2587904"
REQUIRED_LICENSE = "cc-by-4.0"
SPACE = re.compile(r"\s+")


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_receipt(path: Path, **extra: object) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        **extra,
    }


def write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(str(path) + ".partial")
    if partial.exists():
        partial.unlink()
    with partial.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    partial.replace(path)


def fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "GlossAPI-CPT-evaluation/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def download(url: str, output: Path) -> tuple[str, str, int]:
    request = urllib.request.Request(url, headers={"User-Agent": "GlossAPI-CPT-evaluation/1.0"})
    partial = Path(str(output) + ".partial")
    sha256 = hashlib.sha256()
    md5 = hashlib.md5(usedforsecurity=False)
    total = 0
    with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as handle:
        while True:
            chunk = response.read(8 * 1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
            sha256.update(chunk)
            md5.update(chunk)
            total += len(chunk)
        handle.flush()
        os.fsync(handle.fileno())
    partial.replace(output)
    return sha256.hexdigest(), md5.hexdigest(), total


def normalize_header(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().casefold()).strip("_")


def normalized_row(row: dict[str, str]) -> dict[str, str]:
    return {normalize_header(str(key)): str(value or "") for key, value in row.items() if key is not None}


def cluster_fields(row: dict[str, str]) -> dict[str, str]:
    return {
        key: SPACE.sub(" ", row.get(key, "").strip())
        for key in (
            "sitting_date",
            "parliamentary_period",
            "parliamentary_session",
            "parliamentary_sitting",
        )
    }


def cluster_id(row: dict[str, str]) -> str:
    fields = cluster_fields(row)
    if not any(fields.values()):
        raise ValueError("speech row has no sitting-level cluster metadata")
    return "gpp-" + hashlib.sha256(canonical_json(fields).encode("utf-8")).hexdigest()[:24]


def csv_member(archive: zipfile.ZipFile) -> str:
    members = [name for name in archive.namelist() if name.casefold().endswith(".csv") and not name.endswith("/")]
    preferred = [name for name in members if Path(name).name.casefold().startswith("greek_parliament_proceedings")]
    chosen = preferred or members
    if len(chosen) != 1:
        raise ValueError(f"expected exactly one proceedings CSV, found {members}")
    return chosen[0]


def iter_rows(archive_path: Path, member: str) -> Iterable[tuple[int, dict[str, str]]]:
    with zipfile.ZipFile(archive_path) as archive, archive.open(member) as raw:
        text = io.TextIOWrapper(raw, encoding="utf-8-sig", errors="strict", newline="")
        csv.field_size_limit(sys.maxsize)
        reader = csv.DictReader(text)
        normalized_headers = {normalize_header(name) for name in (reader.fieldnames or [])}
        required = {"speech", "sitting_date", "parliamentary_sitting"}
        if not required.issubset(normalized_headers):
            raise ValueError(f"proceedings CSV schema drift: {sorted(normalized_headers)}")
        for row_number, row in enumerate(reader, 1):
            yield row_number, normalized_row(row)


def snapshot_source(args: argparse.Namespace, output_root: Path) -> tuple[dict[str, object], Path, str]:
    api_url = f"https://zenodo.org/api/records/{args.record_id}"
    metadata_bytes = fetch(api_url)
    metadata = json.loads(metadata_bytes)
    if int(metadata.get("id", -1)) != args.record_id:
        raise ValueError("Zenodo record identity drift")
    license_id = str(metadata.get("metadata", {}).get("license", {}).get("id", ""))
    if license_id != REQUIRED_LICENSE:
        raise ValueError(f"source license drift: {license_id}")
    matches = [row for row in metadata.get("files", []) if row.get("key") == args.file_name]
    if len(matches) != 1:
        raise ValueError("source file inventory drift")
    source_file = matches[0]
    if int(source_file.get("size", -1)) != args.expected_bytes:
        raise ValueError("source byte-size drift")
    expected_checksum = f"md5:{args.expected_md5}"
    if source_file.get("checksum") != expected_checksum:
        raise ValueError("source MD5 binding drift")
    url = str(source_file.get("links", {}).get("self", ""))
    if not url.startswith("https://zenodo.org/api/records/"):
        raise ValueError("unexpected source download URL")

    metadata_path = output_root / "zenodo_record_2587904.json"
    write_atomic(metadata_path, metadata_bytes)
    archive_path = output_root / args.file_name
    sha256, md5, total = download(url, archive_path)
    if total != args.expected_bytes or md5 != args.expected_md5:
        raise ValueError("downloaded source archive failed its frozen checksum")
    with zipfile.ZipFile(archive_path) as archive:
        member = csv_member(archive)
        bad = archive.testzip()
        if bad is not None:
            raise ValueError(f"corrupt ZIP member: {bad}")
    snapshot = {
        "schema_version": "apertus_mini_external_source_snapshot_v1",
        "status": "frozen",
        "source_id": SOURCE_ID,
        "record_id": args.record_id,
        "doi": str(metadata.get("doi", "")),
        "publisher": "Zenodo",
        "license": REQUIRED_LICENSE,
        "evaluation_use_authorized": True,
        "api_url": api_url,
        "metadata": file_receipt(metadata_path),
        "archive": file_receipt(archive_path, md5=md5, upstream_bytes=total, upstream_sha256=sha256),
        "csv_member": member,
    }
    snapshot_path = output_root / "source_snapshot.json"
    write_atomic(snapshot_path, (json.dumps(snapshot, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode())
    return snapshot, archive_path, member


def build_reserve(
    args: argparse.Namespace,
    output_root: Path,
    archive_path: Path,
    member: str,
    snapshot: dict[str, object],
) -> dict[str, object]:
    cluster_counts: Counter[str] = Counter()
    cluster_chars: Counter[str] = Counter()
    cluster_metadata: dict[str, dict[str, str]] = {}
    input_rows = 0
    nonempty_rows = 0
    for _, row in iter_rows(archive_path, member):
        input_rows += 1
        speech = row.get("speech", "").strip()
        if not speech:
            continue
        cid = cluster_id(row)
        nonempty_rows += 1
        cluster_counts[cid] += 1
        cluster_chars[cid] += len(speech)
        cluster_metadata.setdefault(cid, cluster_fields(row))
    if len(cluster_counts) < args.reserve_clusters:
        raise ValueError(f"source only contains {len(cluster_counts)} clusters")
    selected = set(
        sorted(
            cluster_counts,
            key=lambda cid: hashlib.sha256(f"neutral-reserve-v1:{cid}".encode()).digest(),
        )[: args.reserve_clusters]
    )

    fragments_path = output_root / "candidate_reserve_fragments.jsonl"
    partial = Path(str(fragments_path) + ".partial")
    selected_rows = 0
    selected_chars = 0
    per_cluster_order: Counter[str] = Counter()
    with partial.open("w", encoding="utf-8") as output:
        for row_number, row in iter_rows(archive_path, member):
            speech = row.get("speech", "").strip()
            if not speech:
                continue
            cid = cluster_id(row)
            if cid not in selected:
                continue
            order = per_cluster_order[cid]
            per_cluster_order[cid] += 1
            payload = {
                "cluster_id": cid,
                "source_id": SOURCE_ID,
                "source_doc_id": f"{cid}:speech:{order:06d}",
                "source_row": row_number,
                "fragment_order": order,
                "sitting": cluster_metadata[cid],
                "text": speech,
            }
            output.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
            selected_rows += 1
            selected_chars += len(speech)
        output.flush()
        os.fsync(output.fileno())
    partial.replace(fragments_path)
    if selected_rows != sum(cluster_counts[cid] for cid in selected):
        raise ValueError("reserve row closure failed")

    cluster_manifest = {
        "schema_version": "apertus_mini_neutral_cluster_reserve_v1",
        "status": "frozen",
        "source_id": SOURCE_ID,
        "source_snapshot": snapshot,
        "cluster_unit": "complete_parliamentary_sitting",
        "fragment_unit": "speech_fragment",
        "selection": "lowest_sha256_neutral_reserve_v1_cluster_ids",
        "source_counts": {
            "csv_rows": input_rows,
            "nonempty_speech_rows": nonempty_rows,
            "document_clusters": len(cluster_counts),
        },
        "reserve_counts": {
            "speech_fragments": selected_rows,
            "document_clusters": len(selected),
            "text_characters": selected_chars,
        },
        "clusters": [
            {
                "cluster_id": cid,
                "speech_fragments": cluster_counts[cid],
                "text_characters": cluster_chars[cid],
                "sitting": cluster_metadata[cid],
            }
            for cid in sorted(selected)
        ],
        "candidate_fragments": file_receipt(fragments_path, rows=selected_rows),
    }
    manifest_path = output_root / "candidate_reserve_manifest.json"
    write_atomic(
        manifest_path,
        (json.dumps(cluster_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(),
    )
    receipt = {
        "schema_version": "apertus_mini_neutral_source_preparation_receipt_v1",
        "status": "passed",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_snapshot": file_receipt(output_root / "source_snapshot.json"),
        "cluster_manifest": file_receipt(manifest_path),
        "candidate_fragments": file_receipt(fragments_path, rows=selected_rows),
        "document_cluster_split": True,
        "candidate_documents_never_used_for_training": True,
    }
    receipt_path = output_root / "source_preparation_receipt.json"
    write_atomic(receipt_path, (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode())
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--record-id", type=int, default=DEFAULT_RECORD_ID)
    parser.add_argument("--file-name", default=DEFAULT_FILE_NAME)
    parser.add_argument("--expected-bytes", type=int, default=DEFAULT_FILE_BYTES)
    parser.add_argument("--expected-md5", default=DEFAULT_FILE_MD5)
    parser.add_argument("--reserve-clusters", type=int, default=DEFAULT_RESERVE_CLUSTERS)
    args = parser.parse_args()
    if args.record_id != DEFAULT_RECORD_ID or args.file_name != DEFAULT_FILE_NAME:
        raise ValueError("this frozen producer only supports Zenodo record 2587904")
    if args.expected_md5 != DEFAULT_FILE_MD5 or args.expected_bytes != DEFAULT_FILE_BYTES:
        raise ValueError("upstream file binding drift")
    if not 64 <= args.reserve_clusters <= 2048:
        raise ValueError("reserve clusters must be between 64 and 2048")
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(output_root)
    output_root.mkdir(parents=True)
    snapshot, archive_path, member = snapshot_source(args, output_root)
    receipt = build_reserve(args, output_root, archive_path, member, snapshot)
    print(json.dumps({"ok": True, "output_root": str(output_root), **receipt["candidate_fragments"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
