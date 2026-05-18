"""Unit test for firing_count_worker — synthetic data, no cloud.

Validates:
  1. Worker runs end-to-end on a synthetic shard
  2. share-sum invariant: per-(source,id) counts sum to component counts
     and to total counts for each id
  3. Per-component row counts are accurately tracked
  4. fail-hard on row-misaligned manifest (fewer rows in manifest)
  5. fail-hard on missing source_dataset column

Run from this scripts/ dir:
    python test_firing_count_worker.py
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
TOKENIZER_DIR = (
    HERE.parent / "variants/c3_added_17408_curated_padded"
)
WORKER = HERE / "firing_count_worker.py"


# ----------------------------------------------------------------------------
# Synthetic data builder
# ----------------------------------------------------------------------------

SYNTHETIC_TEXTS = [
    ("Apothetirio_Kallipos", "Καλημέρα, ο κόσμος είναι μεγάλος. " * 4),
    ("Apothetirio_Kallipos", "Το θέμα μας είναι σήμερα η εκπαίδευση."),
    ("greek_phd", "Η νεοελληνική γλώσσα είναι όμορφη και πλούσια."),
    ("greek_phd", "Ο νεοδιοριζόμενος υπουργός ψηφίζει νέους νόμους."),
    ("greek_phd", "Δοκιμή για τα θέματα μου σήμερα."),
    ("HuggingFaceFW/finewiki", "Η Αθήνα είναι η πρωτεύουσα της Ελλάδας."),
    ("HPLT/ell_Grek_ge8_no_mt_clean60", "Ένα τυχαίο HPLT έγγραφο για δοκιμή."),
    ("HPLT/ell_Grek_ge8_no_mt_clean60", "Δεύτερο HPLT έγγραφο. " * 3),
    ("HPLT/ell_Grek_ge8_no_mt_clean60", "Τρίτο HPLT έγγραφο."),
    ("AI-team-UoA/greek_legal_code", "Ο νόμος ορίζει την υποχρέωση."),
]


def build_synthetic_shard(out_dir: Path) -> tuple[Path, Path]:
    """Build paired (text.parquet, manifest.csv) for testing."""
    out_dir.mkdir(parents=True, exist_ok=True)

    text_path = out_dir / "shard_text.parquet"
    manifest_path = out_dir / "shard_manifest.csv"

    texts = [t for _, t in SYNTHETIC_TEXTS]
    sources = [s for s, _ in SYNTHETIC_TEXTS]

    pq.write_table(
        pa.table({"text": pa.array(texts, type=pa.string())}),
        str(text_path),
    )
    with manifest_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["source_dataset"])
        w.writeheader()
        for s in sources:
            w.writerow({"source_dataset": s})
    return text_path, manifest_path


def build_misaligned_manifest(out_dir: Path, n_drop: int) -> Path:
    """Manifest shorter than text parquet by n_drop rows (should fail-hard)."""
    sources = [s for s, _ in SYNTHETIC_TEXTS[:-n_drop]]
    manifest_path = out_dir / "shard_manifest_short.csv"
    with manifest_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["source_dataset"])
        w.writeheader()
        for s in sources:
            w.writerow({"source_dataset": s})
    return manifest_path


def build_missing_column_manifest(out_dir: Path) -> Path:
    """Manifest with a different column (should fail-hard)."""
    manifest_path = out_dir / "shard_manifest_no_source.csv"
    with manifest_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["doc_id"])
        w.writeheader()
        for i in range(len(SYNTHETIC_TEXTS)):
            w.writerow({"doc_id": str(i)})
    return manifest_path


# ----------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------

def _run_worker(text_parquet: Path, manifest: Path, out_dir: Path,
                local_gcs: Path) -> tuple[int, str, str]:
    """Run the worker against a *local* output dir (fake GCS).

    We monkey-patch the gsutil upload by replacing gcloud with a script
    that just copies the file locally — see _make_gcloud_stub below.
    """
    env = os.environ.copy()
    env["PATH"] = str(out_dir) + ":" + env["PATH"]  # gcloud stub first
    proc = subprocess.run(
        [
            sys.executable, str(WORKER),
            "--text-parquet", str(text_parquet),
            "--manifest", str(manifest),
            "--tokenizer-dir", str(TOKENIZER_DIR),
            "--shard", "0", "--total", "1",
            "--gcs-out-prefix", f"local://{local_gcs}",
            "--batch-size", "4",  # small to exercise multi-batch logic
            "--local-tmp", str(out_dir),
        ],
        capture_output=True, text=True, env=env, cwd=str(out_dir),
    )
    return proc.returncode, proc.stdout, proc.stderr


def _make_gcloud_stub(out_dir: Path) -> None:
    """Drop a shell stub named `gcloud` in out_dir that mimics
    `gcloud storage cp SRC gs://...` → local cp into a fake GCS dir."""
    stub = out_dir / "gcloud"
    stub.write_text(r"""#!/usr/bin/env bash
# Test stub for gcloud — handles 'gcloud storage cp SRC DST'.
# DST starts with local:// or gs://; we mangle into a local filesystem path.
set -e
if [ "$1" == "storage" ] && [ "$2" == "cp" ]; then
    SRC="$3"
    DST="$4"
    # Strip local:// or gs:// prefix
    DST_LOCAL="${DST#local://}"
    DST_LOCAL="${DST_LOCAL#gs://}"
    mkdir -p "$(dirname "$DST_LOCAL")"
    cp "$SRC" "$DST_LOCAL"
    exit 0
fi
echo "test gcloud stub: unhandled args: $@" >&2
exit 99
""")
    stub.chmod(0o755)


def test_happy_path() -> None:
    print("== test_happy_path ==")
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        text_p, manifest_p = build_synthetic_shard(td)
        local_gcs = td / "fake_gcs"
        local_gcs.mkdir()
        _make_gcloud_stub(td)
        rc, stdout, stderr = _run_worker(text_p, manifest_p, td, local_gcs)
        if rc != 0:
            print("STDOUT:", stdout)
            print("STDERR:", stderr)
            raise AssertionError(f"worker exited {rc}")
        # Outputs should exist
        counts_p = local_gcs / "per_source_counts/shard_00_of_01.parquet"
        denoms_p = local_gcs / "per_source_denominators/shard_00_of_01.json"
        done_p = local_gcs / "_DONE_shard_00_of_01"
        assert counts_p.exists(), f"missing {counts_p}"
        assert denoms_p.exists(), f"missing {denoms_p}"
        assert done_p.exists(), f"missing {done_p}"

        # Validate counts schema + share-sum invariant
        tbl = pq.read_table(counts_p)
        assert tbl.column_names == ["source_dataset", "id", "fire_count"]
        df = tbl.to_pandas()
        n_sources = df["source_dataset"].nunique()
        n_ids = df["id"].nunique()
        print(f"  counts rows: {len(df):,}  sources: {n_sources}  ids: {n_ids}")

        # Build per-id total (sum across sources)
        per_id_total = df.groupby("id")["fire_count"].sum()
        # And per-id by component
        df["component"] = df["source_dataset"].apply(
            lambda s: "hplt_only" if s == "HPLT/ell_Grek_ge8_no_mt_clean60"
                       else "glossapi_nanochat_only"
        )
        per_id_component = df.groupby(["component", "id"])["fire_count"].sum().unstack(0).fillna(0)
        # Invariant: per_id_total == component_sums.sum
        component_sums = per_id_component.sum(axis=1)
        for i in per_id_total.index:
            assert per_id_total[i] == component_sums[i], \
                f"share-sum invariant failed for id={i}: total={per_id_total[i]} components_sum={component_sums[i]}"
        print(f"  ✓ share-sum invariant holds for all {len(per_id_total)} ids")

        # Validate denoms
        d = json.loads(denoms_p.read_text())
        n_sources_in_denoms = len(d)
        print(f"  denoms sources: {n_sources_in_denoms}  e.g.:")
        for s in list(d.keys())[:3]:
            print(f"    {s:<40s} {d[s]}")
        # Sanity: every source in counts should also be in denoms
        for s in df["source_dataset"].unique():
            assert s in d, f"source {s} in counts but missing from denoms"

        # Sanity: total tokens in denoms should equal sum of fire_counts
        total_from_counts = df["fire_count"].sum()
        total_from_denoms = sum(v["tokenized_tokens"] for v in d.values())
        assert total_from_counts == total_from_denoms, \
            f"token total mismatch: counts={total_from_counts} denoms={total_from_denoms}"
        print(f"  ✓ total tokens consistent: {total_from_counts:,}")
        print("  PASS")


def test_misaligned_manifest() -> None:
    print("== test_misaligned_manifest (short manifest) ==")
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        text_p, _ = build_synthetic_shard(td)
        short_manifest = build_misaligned_manifest(td, n_drop=3)
        local_gcs = td / "fake_gcs"; local_gcs.mkdir()
        _make_gcloud_stub(td)
        rc, stdout, stderr = _run_worker(text_p, short_manifest, td, local_gcs)
        assert rc != 0, f"expected non-zero exit on misaligned manifest, got {rc}"
        assert "manifest exhausted" in (stdout + stderr), \
            f"expected 'manifest exhausted' in output, got stdout={stdout!r} stderr={stderr!r}"
        # _DONE must NOT exist on failure
        done_p = local_gcs / "_DONE_shard_00_of_01"
        assert not done_p.exists(), "_DONE marker should not exist on failure"
        print("  ✓ worker failed-hard on short manifest, no _DONE written")
        print("  PASS")


def test_missing_source_column() -> None:
    print("== test_missing_source_column ==")
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        text_p, _ = build_synthetic_shard(td)
        bad_manifest = build_missing_column_manifest(td)
        local_gcs = td / "fake_gcs"; local_gcs.mkdir()
        _make_gcloud_stub(td)
        rc, stdout, stderr = _run_worker(text_p, bad_manifest, td, local_gcs)
        assert rc != 0, f"expected non-zero exit on missing source column, got {rc}"
        combined = stdout + stderr
        assert "source_dataset" in combined, \
            f"expected 'source_dataset' in error output: {combined[-500:]!r}"
        print("  ✓ worker failed-hard on missing source_dataset column")
        print("  PASS")


def test_smoke_mode() -> None:
    print("== test_smoke_mode ==")
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        text_p, manifest_p = build_synthetic_shard(td)
        local_gcs = td / "fake_gcs"; local_gcs.mkdir()
        _make_gcloud_stub(td)
        env = os.environ.copy()
        env["PATH"] = str(td) + ":" + env["PATH"]
        proc = subprocess.run(
            [
                sys.executable, str(WORKER),
                "--text-parquet", str(text_p),
                "--manifest", str(manifest_p),
                "--tokenizer-dir", str(TOKENIZER_DIR),
                "--shard", "0", "--total", "1",
                "--gcs-out-prefix", f"local://{local_gcs}",
                "--smoke",
                "--max-total-rows-per-component", "2",
                "--batch-size", "4",
                "--local-tmp", str(td),
            ],
            capture_output=True, text=True, env=env, cwd=str(td),
        )
        if proc.returncode != 0:
            print("STDOUT:", proc.stdout); print("STDERR:", proc.stderr)
            raise AssertionError(f"worker (smoke) exited {proc.returncode}")
        # Smoke outputs in _smoke/
        bench_p = local_gcs / "_smoke/smoke_benchmark.json"
        assert bench_p.exists(), f"missing {bench_p}"
        b = json.loads(bench_p.read_text())
        print(f"  benchmark: tokens/s={b['tokenize_tokens_per_sec']:,}  rows={b['n_rows_processed']}")
        assert b["n_rows_processed"] > 0
        # Reviewer round 3 #6: smoke MUST validate source accounting
        assert "source_datasets_seen" in b, "missing source_datasets_seen in benchmark"
        assert "source_dataset_rows" in b, "missing source_dataset_rows in benchmark"
        assert "component_row_counts" in b, "missing component_row_counts in benchmark"
        seen = b["source_datasets_seen"]
        assert len(seen) > 0, "no source_datasets seen"
        # Sanity: HPLT source string should be one of them given our synthetic data
        assert any("HPLT" in s for s in seen), \
            f"expected at least one HPLT source in {seen}"
        crc = b["component_row_counts"]
        assert crc.get("glossapi_nanochat_only", 0) > 0
        assert crc.get("hplt_only", 0) > 0
        print(f"  ✓ smoke validated source accounting: "
              f"{len(seen)} sources, "
              f"{crc['glossapi_nanochat_only']} GA + {crc['hplt_only']} HPLT rows")
        print("  PASS")


def main() -> int:
    if not (TOKENIZER_DIR / "tokenizer.json").exists():
        print(f"SKIP: tokenizer not found at {TOKENIZER_DIR}")
        print("  (this test needs the c3_added_17408_curated_padded tokenizer)")
        return 0

    tests = [
        test_happy_path,
        test_misaligned_manifest,
        test_missing_source_column,
        test_smoke_mode,
    ]
    fails = 0
    for t in tests:
        try:
            t()
        except Exception as e:
            print(f"  FAIL: {type(e).__name__}: {e}")
            fails += 1
    print()
    print(f"=== {len(tests) - fails}/{len(tests)} passed ===")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
