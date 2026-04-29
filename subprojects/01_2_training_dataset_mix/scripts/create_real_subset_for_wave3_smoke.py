#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


DEFAULT_SPECS = [
    ("1000_prwta_xronia_ellhnikhs.parquet", 400),
    ("AI-team-UoA__greek_legal_code.parquet", 400),
    ("Apothetirio_Kallipos.parquet", 400),
    ("HPLT__ell_Grek_ge8_no_mt_clean60.8_1.part-00000.parquet", 1200),
]


def parse_spec(raw: str) -> tuple[str, int]:
    name, sep, limit = raw.partition(":")
    if not sep:
        raise ValueError(f"Expected FILE:LIMIT spec, got {raw!r}")
    return name, int(limit)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a small real-doc source subset for wave-3 integration tests.")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--spec", action="append", default=None, help="Input parquet basename and row limit: FILE:LIMIT.")
    parser.add_argument("--greek-badness-lt", type=float, default=50.0)
    parser.add_argument("--mojibake-lte", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=2048)
    return parser.parse_args()


def filter_table(table: pa.Table, *, greek_badness_lt: float, mojibake_lte: float) -> pa.Table:
    greek_score = pc.cast(table["greek_badness_score"], pa.float64(), safe=False)
    mojibake_score = pc.cast(table["mojibake_badness_score"], pa.float64(), safe=False)
    greek_ok = pc.fill_null(pc.less(greek_score, greek_badness_lt), False)
    moji_ok = pc.fill_null(pc.less_equal(mojibake_score, mojibake_lte), False)
    if "is_empty" in table.column_names and not pa.types.is_null(table["is_empty"].type):
        non_empty = pc.equal(pc.fill_null(table["is_empty"], False), False)
    else:
        non_empty = pa.array([True] * table.num_rows)
    return table.filter(pc.and_(pc.and_(greek_ok, moji_ok), non_empty))


def build_one(
    input_path: Path,
    output_path: Path,
    *,
    limit: int,
    greek_badness_lt: float,
    mojibake_lte: float,
    batch_size: int,
) -> dict[str, object]:
    pf = pq.ParquetFile(input_path)
    chunks: list[pa.Table] = []
    selected = 0
    for batch in pf.iter_batches(batch_size=batch_size):
        table = pa.Table.from_batches([batch], schema=pf.schema_arrow)
        filtered = filter_table(table, greek_badness_lt=greek_badness_lt, mojibake_lte=mojibake_lte)
        if filtered.num_rows == 0:
            continue
        take = min(limit - selected, filtered.num_rows)
        chunks.append(filtered.slice(0, take))
        selected += take
        if selected >= limit:
            break
    if chunks:
        out_table = pa.concat_tables(chunks, promote_options="default")
    else:
        out_table = pa.Table.from_batches([], schema=pf.schema_arrow)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(out_table, output_path, compression="zstd")
    datasets = out_table["source_dataset"].unique().to_pylist() if out_table.num_rows else []
    chars = int(pc.sum(pc.utf8_length(out_table["text"])).as_py() or 0) if out_table.num_rows else 0
    return {
        "input": str(input_path),
        "output": str(output_path),
        "rows": out_table.num_rows,
        "source_datasets": datasets,
        "chars": chars,
    }


def main() -> None:
    args = parse_args()
    specs = [parse_spec(raw) for raw in args.spec] if args.spec else DEFAULT_SPECS
    data_root = args.output_root / "data"
    data_root.mkdir(parents=True, exist_ok=True)
    summary = []
    for name, limit in specs:
        summary.append(
            build_one(
                args.input_root / name,
                data_root / name,
                limit=limit,
                greek_badness_lt=args.greek_badness_lt,
                mojibake_lte=args.mojibake_lte,
                batch_size=args.batch_size,
            )
        )
    (args.output_root / "source_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
