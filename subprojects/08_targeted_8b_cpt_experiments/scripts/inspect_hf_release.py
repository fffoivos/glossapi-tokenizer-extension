#!/usr/bin/env python3
"""Inspect the anonymized HF release using Parquet metadata only."""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import pyarrow.parquet as pq

from contract_utils import TOKENIZER_SHA256, file_binding, read_json, require, write_json_atomic


EXPECTED_REVISION = "987b8955fcd395c6219e39df9e64715457f69065"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite receipt: {args.output}")
    manifests = args.release_root / "release/manifests"
    anonymization_path = manifests / "anonymization_manifest.json"
    token_counts_path = manifests / "token_counts.json"
    publication_path = args.release_root / "publication/receipt.json"
    public_path = args.release_root / "publication/public_access_receipt.json"
    anonymization = read_json(anonymization_path)
    counts = read_json(token_counts_path)
    publication = read_json(publication_path)
    public = read_json(public_path)
    require(anonymization.get("counts", {}).get("rows") == 51839746, "release row count drift")
    require(anonymization.get("counts", {}).get("shards") == 431, "release shard count drift")
    require(counts.get("status") == "passed", "token counts are not passed")
    require(counts.get("tokenizer", {}).get("tokenizer_json_sha256") == TOKENIZER_SHA256, "release tokenizer drift")
    require(counts.get("totals", {}).get("training_tokens") == 63780757593, "release token total drift")
    require(publication.get("status") == "passed", "publication receipt is not passed")
    require(public.get("status") == "passed", "public access receipt is not passed")
    # Publication receipt layouts have changed; accept the immutable target if
    # it appears in either the top-level or published-revision field.
    revisions = {
        str(publication.get(key, ""))
        for key in ("revision", "published_revision", "commit", "commit_sha")
    }
    revisions.update(str(public.get(key, "")) for key in ("revision", "published_revision", "commit", "commit_sha"))
    require(EXPECTED_REVISION in revisions or EXPECTED_REVISION in publication_path.read_text(), "published HF revision drift")
    expected_files = anonymization.get("files", [])
    require(len(expected_files) == 431, "anonymization file inventory drift")
    schemas: dict[str, int] = {}
    total_rows = 0
    for row in expected_files:
        path = args.release_root / "release" / row["path"]
        require(path.is_file() and path.stat().st_size == int(row["bytes"]), f"Parquet size drift: {path}")
        metadata = pq.ParquetFile(path).metadata
        require(metadata.num_rows == int(row["rows"]), f"Parquet row metadata drift: {path}")
        schema = str(pq.ParquetFile(path).schema_arrow)
        schemas[schema] = schemas.get(schema, 0) + 1
        total_rows += metadata.num_rows
    require(total_rows == 51839746, "Parquet metadata rows do not reconcile")
    require(len(schemas) == 1, f"release has {len(schemas)} distinct Parquet schemas")
    schema_text = next(iter(schemas))
    for required_column in ("text", "source_dataset"):
        require(required_column in schema_text, f"release lacks {required_column} column")
    for source, expected in {"openarchives.gr": 126597, "greek_phd": 31692, "HPLT/ell_Grek_ge8_no_mt_clean60": 48629460}.items():
        require(counts["source_rows"].get(source) == expected, f"source row count drift: {source}")
    payload = {
        "schema_version": "targeted_8b_hf_release_inspection_v1",
        "status": "passed",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "release_root": str(args.release_root.resolve()),
        "bindings": {
            "anonymization_manifest": file_binding(anonymization_path),
            "token_counts": file_binding(token_counts_path),
            "publication": file_binding(publication_path),
            "public_access": file_binding(public_path),
        },
        "parquet": {"files": len(expected_files), "rows": total_rows, "schema": schema_text},
        "source_rows": {source: counts["source_rows"][source] for source in ("openarchives.gr", "greek_phd", "HPLT/ell_Grek_ge8_no_mt_clean60")},
        "source_text_tokens": {source: counts["source_text_tokens"][source] for source in ("openarchives.gr", "greek_phd", "HPLT/ell_Grek_ge8_no_mt_clean60")},
        "hf_revision": EXPECTED_REVISION,
        "no_data_rows_written": True,
    }
    write_json_atomic(args.output, payload)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
