"""End-to-end test: worker × K shards → aggregator → final parquets.

Uses synthetic train shards + manifests; no cloud. Validates:
  1. Aggregator sums per-source counts across K shards correctly
  2. share-sum invariant survives the aggregation
  3. Component derivation matches expectations
  4. Per-source long parquet has the right cardinality
  5. fail-hard if a shard's _DONE marker is missing

Run from this scripts/ dir:
    python test_aggregate_firing_counts.py
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
AGGREGATOR = HERE / "aggregate_firing_counts.py"


# Two shards' worth of synthetic data (split same 10 docs across 2 shards)
SHARD_A_TEXTS = [
    ("Apothetirio_Kallipos", "Καλημέρα, ο κόσμος είναι μεγάλος. " * 4),
    ("Apothetirio_Kallipos", "Το θέμα μας είναι σήμερα η εκπαίδευση."),
    ("greek_phd", "Η νεοελληνική γλώσσα είναι όμορφη και πλούσια."),
    ("greek_phd", "Ο νεοδιοριζόμενος υπουργός ψηφίζει νέους νόμους."),
    ("HPLT/ell_Grek_ge8_no_mt_clean60", "Ένα τυχαίο HPLT έγγραφο για δοκιμή."),
]
SHARD_B_TEXTS = [
    ("greek_phd", "Δοκιμή για τα θέματα μου σήμερα."),
    ("HuggingFaceFW/finewiki", "Η Αθήνα είναι η πρωτεύουσα της Ελλάδας."),
    ("HPLT/ell_Grek_ge8_no_mt_clean60", "Δεύτερο HPLT έγγραφο. " * 3),
    ("HPLT/ell_Grek_ge8_no_mt_clean60", "Τρίτο HPLT έγγραφο."),
    ("AI-team-UoA/greek_legal_code", "Ο νόμος ορίζει την υποχρέωση."),
]


def build_shard_pair(out_dir: Path, tag: str, docs):
    texts = [t for _, t in docs]
    sources = [s for s, _ in docs]
    text_p = out_dir / f"{tag}_text.parquet"
    manifest_p = out_dir / f"{tag}_manifest.csv"
    pq.write_table(pa.table({"text": pa.array(texts, type=pa.string())}), str(text_p))
    with manifest_p.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["source_dataset"])
        w.writeheader()
        for s in sources:
            w.writerow({"source_dataset": s})
    return text_p, manifest_p


def make_gcloud_stub(out_dir: Path) -> None:
    """gcloud cp stub: local copy. Also handles `gcloud storage ls`."""
    stub = out_dir / "gcloud"
    stub.write_text(r"""#!/usr/bin/env bash
set -e
if [ "$1" == "storage" ] && [ "$2" == "cp" ]; then
    SRC="$3"; DST="$4"
    SRC_LOCAL="${SRC#local://}"
    SRC_LOCAL="${SRC_LOCAL#gs://}"
    DST_LOCAL="${DST#local://}"
    DST_LOCAL="${DST_LOCAL#gs://}"
    mkdir -p "$(dirname "$DST_LOCAL")"
    cp "$SRC_LOCAL" "$DST_LOCAL"
    exit 0
fi
if [ "$1" == "storage" ] && [ "$2" == "ls" ]; then
    SRC="$3"
    SRC_LOCAL="${SRC#local://}"
    SRC_LOCAL="${SRC_LOCAL#gs://}"
    # ls all files under this prefix (recursive)
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


def run_worker(text_p, manifest_p, shard_idx, total, gcs_prefix, env_path, cwd):
    env = os.environ.copy()
    env["PATH"] = env_path + ":" + env["PATH"]
    proc = subprocess.run(
        [
            sys.executable, str(WORKER),
            "--text-parquet", str(text_p),
            "--manifest", str(manifest_p),
            "--tokenizer-dir", str(TOKENIZER_DIR),
            "--shard", str(shard_idx), "--total", str(total),
            "--gcs-out-prefix", gcs_prefix,
            "--batch-size", "4",
            "--local-tmp", str(cwd),
        ],
        capture_output=True, text=True, env=env, cwd=str(cwd),
    )
    if proc.returncode != 0:
        print("STDOUT:", proc.stdout); print("STDERR:", proc.stderr)
        raise AssertionError(f"worker shard {shard_idx} exited {proc.returncode}")


def test_end_to_end_aggregation() -> None:
    print("== test_end_to_end_aggregation ==")
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        # Synthetic shard A + B
        ta, ma = build_shard_pair(td, "A", SHARD_A_TEXTS)
        tb, mb = build_shard_pair(td, "B", SHARD_B_TEXTS)
        # Fake GCS root
        local_gcs = td / "fake_gcs"
        local_gcs.mkdir()
        make_gcloud_stub(td)
        env_path = str(td)

        # Run workers
        gcs_prefix = f"local://{local_gcs}"
        run_worker(ta, ma, 0, 2, gcs_prefix, env_path, td)
        run_worker(tb, mb, 1, 2, gcs_prefix, env_path, td)

        # Sanity: both shards should have _DONE markers
        for i in [0, 1]:
            done_p = local_gcs / f"_DONE_shard_{i:02d}_of_02"
            assert done_p.exists(), f"missing {done_p}"

        # Run aggregator
        out_dir = td / "final"
        env = os.environ.copy()
        env["PATH"] = env_path + ":" + env["PATH"]
        proc = subprocess.run(
            [
                sys.executable, str(AGGREGATOR),
                "--gcs-prefix", gcs_prefix,
                "--k", "2",
                "--tokenizer-dir", str(TOKENIZER_DIR),
                "--out-dir", str(out_dir),
            ],
            capture_output=True, text=True, env=env, cwd=str(td),
        )
        if proc.returncode != 0:
            print("STDOUT:", proc.stdout); print("STDERR:", proc.stderr)
            raise AssertionError(f"aggregator exited {proc.returncode}")

        # Validate outputs
        for fname in [
            "source_dataset_token_counts.parquet",  # renamed (reviewer #5)
            "source_dataset_summary.parquet",       # new (reviewer #5)
            "glossapi_nanochat_only.parquet",
            "hplt_only.parquet",
            "glossapi_nanochat_plus_hplt.parquet",
            "run_summary.json",
        ]:
            assert (out_dir / fname).exists(), f"missing {out_dir/fname}"
        print(f"  ✓ all 6 output files exist")

        # Share-sum invariant: per_source totals = combined totals
        ps = pq.read_table(out_dir / "source_dataset_token_counts.parquet").to_pandas()
        ga = pq.read_table(out_dir / "glossapi_nanochat_only.parquet").to_pandas()
        hp = pq.read_table(out_dir / "hplt_only.parquet").to_pandas()
        cb = pq.read_table(out_dir / "glossapi_nanochat_plus_hplt.parquet").to_pandas()

        ps_total = ps.groupby("id")["fire_count"].sum()
        cb_total = cb.set_index("id")["fire_count"]
        for i in ps_total.index:
            if ps_total[i] != cb_total[i]:
                raise AssertionError(
                    f"share-sum invariant failed at id={i}: "
                    f"per_source_total={ps_total[i]} combined={cb_total[i]}"
                )
        print(f"  ✓ share-sum invariant holds for {len(ps_total)} ids")

        # Component invariant: glossapi + hplt = combined (every id)
        comb_check = ga.set_index("id")["fire_count"] + hp.set_index("id")["fire_count"]
        cb_total_full = cb.set_index("id")["fire_count"]
        assert (comb_check == cb_total_full).all(), \
            "component invariant violated: ga + hp != combined"
        print(f"  ✓ component invariant: ga + hp = combined")

        # Per-source has both source_dataset and source_group columns
        cols = pq.read_table(out_dir / "source_dataset_token_counts.parquet").column_names
        for c in ["source_dataset", "source_group", "id", "decoded",
                  "fire_count", "fire_rate_within_source_dataset",
                  "share_of_component_token_firings"]:
            assert c in cols, f"source_dataset_token_counts missing column {c}"
        print(f"  ✓ source_dataset_token_counts has all 7 expected columns")

        # Vocab-size check (reviewer round 3 #1): every component vector
        # must have len == tokenizer vocab_size, not max(observed id) + 1
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(str(TOKENIZER_DIR / "tokenizer.json"))
        expected_vocab = tok.get_vocab_size()
        ga_full = pq.read_table(out_dir / "glossapi_nanochat_only.parquet").to_pandas()
        assert len(ga_full) == expected_vocab, \
            f"glossapi_nanochat_only has {len(ga_full)} rows, expected {expected_vocab} (tokenizer vocab)"
        print(f"  ✓ component parquets cover full vocab ({expected_vocab:,} ids each — incl zero-firing tail)")

        # Summary JSON sanity
        summary = json.loads((out_dir / "run_summary.json").read_text())
        assert summary["k_shards"] == 2
        assert summary["n_distinct_sources"] == 5
        n_hplt_rows = summary["component_denominators"]["hplt_only"]["rows"]
        n_ga_rows = summary["component_denominators"]["glossapi_nanochat_only"]["rows"]
        assert n_hplt_rows + n_ga_rows == 10, \
            f"total docs should be 10, got {n_hplt_rows + n_ga_rows}"
        print(f"  ✓ summary: 5 sources, {n_ga_rows} GA rows + {n_hplt_rows} HPLT rows = 10")
        print("  PASS")


def test_missing_done_marker_fails_hard() -> None:
    print("== test_missing_done_marker_fails_hard ==")
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        ta, ma = build_shard_pair(td, "A", SHARD_A_TEXTS)
        tb, mb = build_shard_pair(td, "B", SHARD_B_TEXTS)
        local_gcs = td / "fake_gcs"; local_gcs.mkdir()
        make_gcloud_stub(td)
        env_path = str(td)
        gcs_prefix = f"local://{local_gcs}"
        run_worker(ta, ma, 0, 2, gcs_prefix, env_path, td)
        # NOTE: deliberately do NOT run shard 1
        out_dir = td / "final"
        env = os.environ.copy()
        env["PATH"] = env_path + ":" + env["PATH"]
        proc = subprocess.run(
            [
                sys.executable, str(AGGREGATOR),
                "--gcs-prefix", gcs_prefix,
                "--k", "2",
                "--tokenizer-dir", str(TOKENIZER_DIR),
                "--out-dir", str(out_dir),
            ],
            capture_output=True, text=True, env=env, cwd=str(td),
        )
        assert proc.returncode != 0, \
            f"aggregator should fail on missing _DONE; exit={proc.returncode}"
        combined = proc.stdout + proc.stderr
        assert "Missing _DONE markers" in combined, \
            f"expected 'Missing _DONE markers' in output: {combined[-400:]!r}"
        print("  ✓ aggregator failed-hard on missing _DONE")
        print("  PASS")


def main() -> int:
    if not (TOKENIZER_DIR / "tokenizer.json").exists():
        print(f"SKIP: tokenizer not found at {TOKENIZER_DIR}")
        return 0
    tests = [test_end_to_end_aggregation, test_missing_done_marker_fails_hard]
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
