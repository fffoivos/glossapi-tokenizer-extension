"""Unit test for build_shard_manifests.py — synthetic multi-row-group
text + manifest, sharded into K paired files, then verified by
re-aggregation.

Validates (reviewer round 3 #4):
  1. Sharder writes K paired (text, manifest) files
  2. Total rows across shards == original row count
  3. Per-row text/source alignment is preserved (each text in shard N at
     position i has the same source as the original at the same row index)
  4. Sum of per-shard worker outputs equals direct unsharded counting
     (end-to-end correctness against the sharded pipeline)

Run:
    python test_build_shard_manifests.py
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


HERE = Path(__file__).resolve().parent
TOKENIZER_DIR = HERE.parent / "variants/c3_added_17408_curated_padded"
WORKER = HERE / "firing_count_worker.py"
SHARDER = HERE / "build_shard_manifests.py"
AGGREGATOR = HERE / "aggregate_firing_counts.py"


# Multi-source synthetic corpus split across multiple row groups
DOCS = [
    ("Apothetirio_Kallipos", "Καλημέρα, ο κόσμος είναι μεγάλος. " * i)
    for i in range(1, 16)
] + [
    ("greek_phd", "Η νεοελληνική γλώσσα είναι όμορφη. " * i)
    for i in range(1, 12)
] + [
    ("AI-team-UoA/greek_legal_code", "Ο νόμος ορίζει την υποχρέωση. " * i)
    for i in range(1, 8)
] + [
    ("HuggingFaceFW/finewiki", "Η Αθήνα είναι η πρωτεύουσα. " * i)
    for i in range(1, 6)
] + [
    ("HPLT/ell_Grek_ge8_no_mt_clean60", "HPLT έγγραφο για δοκιμή. " * i)
    for i in range(1, 25)
]


def build_synthetic_corpus(out_dir: Path) -> tuple[Path, Path, int]:
    """Build train.parquet (multi-row-group) + train_manifest.parquet."""
    texts = [t for _, t in DOCS]
    sources = [s for s, _ in DOCS]
    n_rows = len(DOCS)
    text_path = out_dir / "train.parquet"
    manifest_path = out_dir / "train_manifest.parquet"
    # Write text with small row groups to force multi-row-group behavior
    text_table = pa.table({"text": pa.array(texts, type=pa.string())})
    pq.write_table(text_table, str(text_path), row_group_size=10)
    pq.write_table(
        pa.table({"source_dataset": pa.array(sources, type=pa.string())}),
        str(manifest_path), row_group_size=10,
    )
    return text_path, manifest_path, n_rows


def make_gcloud_stub(out_dir: Path) -> None:
    stub = out_dir / "gcloud"
    stub.write_text(r"""#!/usr/bin/env bash
set -e
if [ "$1" == "storage" ] && [ "$2" == "cp" ]; then
    SRC="$3"; DST="$4"
    SRC_LOCAL="${SRC#local://}"; SRC_LOCAL="${SRC_LOCAL#gs://}"
    DST_LOCAL="${DST#local://}"; DST_LOCAL="${DST_LOCAL#gs://}"
    mkdir -p "$(dirname "$DST_LOCAL")"
    cp "$SRC_LOCAL" "$DST_LOCAL"
    exit 0
fi
if [ "$1" == "storage" ] && [ "$2" == "ls" ]; then
    SRC="$3"
    SRC_LOCAL="${SRC#local://}"; SRC_LOCAL="${SRC_LOCAL#gs://}"
    if [ -d "$SRC_LOCAL" ]; then
        find "$SRC_LOCAL" -type f | sed "s|^|local://|"
    elif [ -f "$SRC_LOCAL" ]; then
        echo "local://$SRC_LOCAL"
    fi
    exit 0
fi
echo "stub: unhandled args: $@" >&2
exit 99
""")
    stub.chmod(0o755)


def run_sharder(text_p: Path, manifest_p: Path, k: int, gcs_prefix: str,
                env_path: str, cwd: Path) -> None:
    env = os.environ.copy()
    env["PATH"] = env_path + ":" + env["PATH"]
    proc = subprocess.run(
        [
            sys.executable, str(SHARDER),
            "--text-parquet", str(text_p),
            "--manifest", str(manifest_p),
            "--k", str(k),
            "--gcs-out-prefix", gcs_prefix,
            "--local-tmp", str(cwd / "sharder_tmp"),
        ],
        capture_output=True, text=True, env=env, cwd=str(cwd),
    )
    if proc.returncode != 0:
        print("STDOUT:", proc.stdout); print("STDERR:", proc.stderr)
        raise AssertionError(f"sharder exited {proc.returncode}")


def test_shard_then_aggregate() -> None:
    print("== test_shard_then_aggregate ==")
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        # Synthetic corpus
        text_p, manifest_p, n_rows = build_synthetic_corpus(td)
        print(f"  built corpus: {n_rows} rows, "
              f"{pq.ParquetFile(text_p).num_row_groups} row_groups")

        # Fake GCS
        local_gcs = td / "fake_gcs"; local_gcs.mkdir()
        make_gcloud_stub(td)
        env_path = str(td)

        K = 3
        gcs_prefix = f"local://{local_gcs}"

        # Run sharder
        run_sharder(text_p, manifest_p, K, gcs_prefix, env_path, td)

        # Verify shards
        shards_dir = local_gcs / "shards"
        for i in range(K):
            tag = f"shard_{i:02d}_of_0{K}"
            text_shard = shards_dir / f"{tag}_text.parquet"
            man_shard = shards_dir / f"{tag}_manifest.parquet"
            assert text_shard.exists(), f"missing {text_shard}"
            assert man_shard.exists(), f"missing {man_shard}"
        print(f"  ✓ {K} paired shards written")

        # Total rows across shards == n_rows
        total_rows = 0
        for i in range(K):
            tag = f"shard_{i:02d}_of_0{K}"
            t = pq.read_table(shards_dir / f"{tag}_text.parquet").num_rows
            m = pq.read_table(shards_dir / f"{tag}_manifest.parquet").num_rows
            assert t == m, f"shard {i}: text rows {t} != manifest rows {m}"
            total_rows += t
        assert total_rows == n_rows, \
            f"sum of shard rows {total_rows} != original {n_rows}"
        print(f"  ✓ row count preserved across shards ({n_rows} = sum of shards)")

        # Row alignment: build a (text → source) map from the original, then
        # verify each (text, source) pair in the shards matches.
        original_pairs = list(zip(
            pq.read_table(text_p).column("text").to_pylist(),
            pq.read_table(manifest_p).column("source_dataset").to_pylist(),
        ))
        original_multiset = sorted(original_pairs)
        sharded_pairs = []
        for i in range(K):
            tag = f"shard_{i:02d}_of_0{K}"
            texts = pq.read_table(shards_dir / f"{tag}_text.parquet").column("text").to_pylist()
            srcs = pq.read_table(shards_dir / f"{tag}_manifest.parquet").column("source_dataset").to_pylist()
            assert len(texts) == len(srcs)
            sharded_pairs.extend(zip(texts, srcs))
        sharded_multiset = sorted(sharded_pairs)
        assert original_multiset == sharded_multiset, \
            "sharded (text, source) pairs do NOT match original after re-sort"
        print(f"  ✓ row alignment preserved (every (text, source) pair survives sharding)")

        # End-to-end: run worker on each shard, then aggregator, then
        # compare to a baseline single-shard worker run on the original.
        # Both should produce the same final counts.
        out_sharded = td / "out_sharded"
        out_sharded.mkdir()
        env = os.environ.copy()
        env["PATH"] = env_path + ":" + env["PATH"]
        for i in range(K):
            tag = f"shard_{i:02d}_of_0{K}"
            proc = subprocess.run(
                [sys.executable, str(WORKER),
                 "--text-parquet", str(shards_dir / f"{tag}_text.parquet"),
                 "--manifest", str(shards_dir / f"{tag}_manifest.parquet"),
                 "--tokenizer-dir", str(TOKENIZER_DIR),
                 "--shard", str(i), "--total", str(K),
                 "--gcs-out-prefix", gcs_prefix,
                 "--batch-size", "4",
                 "--local-tmp", str(td)],
                capture_output=True, text=True, env=env, cwd=str(td),
            )
            if proc.returncode != 0:
                print(proc.stdout); print(proc.stderr)
                raise AssertionError(f"worker shard {i} exited {proc.returncode}")
        proc = subprocess.run(
            [sys.executable, str(AGGREGATOR),
             "--gcs-prefix", gcs_prefix,
             "--k", str(K),
             "--tokenizer-dir", str(TOKENIZER_DIR),
             "--out-dir", str(out_sharded)],
            capture_output=True, text=True, env=env, cwd=str(td),
        )
        if proc.returncode != 0:
            print(proc.stdout); print(proc.stderr)
            raise AssertionError(f"aggregator exited {proc.returncode}")
        sharded_summary = json.loads((out_sharded / "run_summary.json").read_text())

        # Baseline: K=1 against the unsharded original
        local_gcs_baseline = td / "fake_gcs_baseline"; local_gcs_baseline.mkdir()
        baseline_prefix = f"local://{local_gcs_baseline}"
        proc = subprocess.run(
            [sys.executable, str(WORKER),
             "--text-parquet", str(text_p),
             "--manifest", str(manifest_p),
             "--tokenizer-dir", str(TOKENIZER_DIR),
             "--shard", "0", "--total", "1",
             "--gcs-out-prefix", baseline_prefix,
             "--batch-size", "4",
             "--local-tmp", str(td)],
            capture_output=True, text=True, env=env, cwd=str(td),
        )
        if proc.returncode != 0:
            print(proc.stdout); print(proc.stderr)
            raise AssertionError(f"baseline worker exited {proc.returncode}")
        out_baseline = td / "out_baseline"
        out_baseline.mkdir()
        proc = subprocess.run(
            [sys.executable, str(AGGREGATOR),
             "--gcs-prefix", baseline_prefix,
             "--k", "1",
             "--tokenizer-dir", str(TOKENIZER_DIR),
             "--out-dir", str(out_baseline)],
            capture_output=True, text=True, env=env, cwd=str(td),
        )
        if proc.returncode != 0:
            print(proc.stdout); print(proc.stderr)
            raise AssertionError(f"baseline aggregator exited {proc.returncode}")
        baseline_summary = json.loads((out_baseline / "run_summary.json").read_text())

        # Compare component totals — must match exactly
        for comp in ["glossapi_nanochat_only", "hplt_only"]:
            s = sharded_summary["component_denominators"][comp]
            b = baseline_summary["component_denominators"][comp]
            for fld in ["rows", "chars", "tokenized_tokens"]:
                assert s[fld] == b[fld], \
                    f"{comp}.{fld}: sharded={s[fld]} baseline={b[fld]}"
        print(f"  ✓ sharded run matches baseline (rows, chars, tokens per component)")

        # Compare per-source aggregates
        s_per_src = sharded_summary["per_source_summary"]
        b_per_src = baseline_summary["per_source_summary"]
        assert set(s_per_src.keys()) == set(b_per_src.keys()), \
            f"source set differs: sharded={set(s_per_src.keys())} baseline={set(b_per_src.keys())}"
        for src in s_per_src:
            for fld in ["rows", "chars", "tokenized_tokens", "total_added_firings"]:
                assert s_per_src[src][fld] == b_per_src[src][fld], \
                    f"{src}.{fld}: sharded={s_per_src[src][fld]} baseline={b_per_src[src][fld]}"
        print(f"  ✓ per-source counts match baseline for all {len(s_per_src)} sources")

        # Compare final component count vectors directly (the canonical artifacts)
        for comp in ["glossapi_nanochat_only", "hplt_only", "glossapi_nanochat_plus_hplt"]:
            s_tbl = pq.read_table(out_sharded / f"{comp}.parquet")
            b_tbl = pq.read_table(out_baseline / f"{comp}.parquet")
            s_counts = s_tbl.column("fire_count").to_numpy()
            b_counts = b_tbl.column("fire_count").to_numpy()
            assert np.array_equal(s_counts, b_counts), \
                f"{comp}: fire_count vectors differ between sharded and baseline"
        print(f"  ✓ component count vectors bit-identical (sharded == baseline)")
        print("  PASS")


def main() -> int:
    if not (TOKENIZER_DIR / "tokenizer.json").exists():
        print(f"SKIP: tokenizer not found at {TOKENIZER_DIR}")
        return 0
    try:
        test_shard_then_aggregate()
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"  FAIL: {type(e).__name__}: {e}")
        return 1
    print()
    print("=== 1/1 passed ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
