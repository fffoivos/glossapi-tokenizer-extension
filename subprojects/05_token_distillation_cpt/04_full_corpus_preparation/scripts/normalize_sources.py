#!/usr/bin/env python3
"""Normalize receipt-bound Phase-04 sources into canonical Parquet shards."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import tempfile
from collections import Counter
from pathlib import Path

from full_corpus_io import (
    artifacts_from_receipt,
    base_family_map,
    canonical_row,
    canonical_schema,
    expand_nested_row,
    iter_artifact_rows,
    iter_grouped_section_rows,
    sha256_text,
)
from source_lineage import load_json


def write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


class ShardWriter:
    def __init__(self, root: Path, source_id: str, rows_per_shard: int) -> None:
        import pyarrow.parquet as pq

        self.pq = pq
        self.root = root / source_id
        self.root.mkdir(parents=True, exist_ok=True)
        self.rows_per_shard = rows_per_shard
        self.schema = canonical_schema()
        self.rows: list[dict] = []
        self.shard = 0
        self.paths: list[Path] = []

    def add(self, row: dict) -> None:
        self.rows.append(row)
        if len(self.rows) >= self.rows_per_shard:
            self.flush()

    def flush(self) -> None:
        if not self.rows:
            return
        import pyarrow as pa

        destination = self.root / f"part-{self.shard:05d}.parquet"
        temporary = destination.with_suffix(".parquet.partial")
        table = pa.Table.from_pylist(self.rows, schema=self.schema)
        self.pq.write_table(table, temporary, compression="zstd", row_group_size=min(8192, len(self.rows)))
        os.replace(temporary, destination)
        self.paths.append(destination)
        self.rows = []
        self.shard += 1

    def close(self) -> list[Path]:
        self.flush()
        return self.paths


def main() -> int:
    here = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", type=Path, default=here / "configs" / "sources.json")
    parser.add_argument(
        "--lineage-aliases",
        type=Path,
        default=here / "configs" / "source_lineage_aliases.json",
    )
    parser.add_argument("--acquisition-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source", action="append")
    parser.add_argument("--rows-per-shard", type=int, default=50_000)
    parser.add_argument("--max-rows-per-source", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=100_000)
    args = parser.parse_args()
    if args.rows_per_shard < 1:
        parser.error("--rows-per-shard must be positive")
    if args.manifest.exists():
        raise FileExistsError(f"refusing to overwrite immutable manifest: {args.manifest}")
    selected = set(args.source or []) or None
    artifacts = artifacts_from_receipt(args.sources, args.acquisition_receipt, selected)
    lineage_aliases = load_json(args.lineage_aliases)
    source_registry = load_json(args.sources)
    base_families = base_family_map(source_registry, lineage_aliases)
    args.output.mkdir(parents=True, exist_ok=True)
    summaries: list[dict] = []
    all_uids: set[str] = set()
    for artifact in artifacts:
        writer = ShardWriter(args.output, artifact.source_id, args.rows_per_shard)
        counts: Counter[str] = Counter()
        source_names: Counter[str] = Counter()
        work_counts: Counter[str] = Counter()
        for path in artifact.files:
            grouped_sections = "group_sections_to_work" in str(artifact.config.get("merge_policy", ""))
            if grouped_sections:
                source_rows = (
                    (row_index, raw_row, [("0", text_field, raw_text)])
                    for row_index, raw_row, text_field, raw_text in iter_grouped_section_rows(
                        artifact,
                        path,
                        temporary_root=args.output / ".section-spool",
                    )
                )
            else:
                source_rows = (
                    (row_index, raw_row, list(expand_nested_row(raw_row, artifact)))
                    for row_index, raw_row in iter_artifact_rows(path)
                )
            for row_index, raw_row, representations in source_rows:
                counts["rows_scanned"] += 1
                emitted_this_row = 0
                for suffix, text_field, raw_text in representations:
                    row = canonical_row(
                        source=artifact,
                        artifact_path=path,
                        artifact_row_index=row_index,
                        raw_row=raw_row,
                        representation_suffix=suffix,
                        text_field=text_field,
                        raw_text=raw_text,
                        lineage_aliases=lineage_aliases,
                        base_families=base_families,
                    )
                    if row["stable_uid"] in all_uids:
                        raise ValueError(f"duplicate stable_uid during normalization: {row['stable_uid']}")
                    all_uids.add(row["stable_uid"])
                    writer.add(row)
                    emitted_this_row += 1
                    counts["documents_emitted"] += 1
                    counts["characters"] += len(row["text"])
                    counts["bytes_utf8"] += len(row["text"].encode("utf-8"))
                    counts["empty_documents"] += int(not row["text"])
                    source_names[row["source_dataset"]] += 1
                    work_counts[row["work_key"]] += 1
                    if args.max_rows_per_source and counts["documents_emitted"] >= args.max_rows_per_source:
                        break
                if not emitted_this_row:
                    counts["rows_without_text"] += 1
                if args.max_rows_per_source and counts["documents_emitted"] >= args.max_rows_per_source:
                    break
                if args.progress_every and counts["rows_scanned"] % args.progress_every == 0:
                    print(
                        f"normalize_sources: source={artifact.source_id} scanned={counts['rows_scanned']:,} "
                        f"emitted={counts['documents_emitted']:,}",
                        flush=True,
                    )
            if args.max_rows_per_source and counts["documents_emitted"] >= args.max_rows_per_source:
                break
        paths = writer.close()
        summaries.append(
            {
                "source_id": artifact.source_id,
                "repo_id": artifact.repo_id,
                "revision": artifact.revision,
                "source_family_id": artifact.source_family_id,
                "role": artifact.role,
                "counts": dict(counts),
                "exact_source_dataset_counts": dict(sorted(source_names.items())),
                "unique_work_keys": len(work_counts),
                "multi_representation_work_keys": sum(value > 1 for value in work_counts.values()),
                "shards": [str(path) for path in paths],
            }
        )
        print(f"normalize_sources: completed {artifact.source_id}: {counts['documents_emitted']:,} docs", flush=True)
    payload = {
        "schema_version": "full_cpt_normalization_manifest_v1",
        "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "sources_config": str(args.sources.resolve()),
        "sources_config_sha256": sha256_text(args.sources.read_text(encoding="utf-8")),
        "lineage_aliases": str(args.lineage_aliases.resolve()),
        "lineage_aliases_sha256": sha256_text(args.lineage_aliases.read_text(encoding="utf-8")),
        "acquisition_receipt": str(args.acquisition_receipt.resolve()),
        "acquisition_receipt_sha256": sha256_text(args.acquisition_receipt.read_text(encoding="utf-8")),
        "output": str(args.output.resolve()),
        "bounded_smoke": bool(args.max_rows_per_source),
        "sources": summaries,
        "total_documents": sum(row["counts"].get("documents_emitted", 0) for row in summaries),
    }
    write_json_atomic(args.manifest, payload)
    print(f"wrote {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
