#!/usr/bin/env python3
"""Stage 1 — validate & characterize the DEDUPED dataset (SELECTED).

The deduped artifact already exists (SELECTED, produced by
`glossapi_corpus_cli mix-prepare-selected-input`). This does NOT re-derive it;
it PROVES it is the correct deduped artifact and records a provenance manifest.

Hardened after adversarial codex review (see stage1_validate_deduped.codex_review.md).
Guards against vacuous PASS: every check is anchored to an EXPECTED value
(drop-list size, nanochat source totals, Apertus-overlap count), the key-space
is type-checked, membership uses Arrow `is_in` (vectorized), the manifest is
written atomically, and provenance records exact inputs + tool versions.

Checks
  - drop-list integrity: rows / nulls / empties / distinct == --expect-drop-keys
  - key-space soundness: SELECTED key col type == drop-list key col type
  - apertus-overlap applied: # SELECTED keys ∈ drop-list == 0 (only meaningful
    if key-space is sound — reported tri-state)
  - data-loss guard: SELECTED rows ≤ (nanochat_total − apertus_overlap) and > 0;
    per-source breakdown vs expected nanochat source totals; implied internal-
    dedup drops reported
  - optional internal-dedup distinct-key check (--check-dup-keys)

Library: pyarrow (dataset + compute). Run on a compute node.
  python stage1_validate_deduped.py \
    --selected  /iopsstor/.../selected_after_apertus_and_internal_dedup.parquet \
    --drop-list /iopsstor/.../cpt_final_overlay/apertus_overlap_drop_docs.parquet \
    --selected-key-col <K> --drop-key-col <K> --source-col source_dataset \
    --output-manifest deduped_dataset_manifest.json
"""
from __future__ import annotations
import argparse, collections, json, os, socket, subprocess, sys, time

import pyarrow as pa
import pyarrow.dataset as pads
import pyarrow.compute as pc
import pyarrow.parquet as pq

# Expected source totals in the nanochat pool feeding the dedup (from the audit
# concat_summary.json, run dedup_20260519T010924Z). Overridable.
EXPECT = {
    "nanochat_hplt_clean60": 48_728_774,
    "nanochat_glossapi": 49_474_947,
    "apertus_overlap_unique": 2_223_742,
}
EXPECT["nanochat_total"] = EXPECT["nanochat_hplt_clean60"] + EXPECT["nanochat_glossapi"]
EXPECT["selected_rows_upper_bound"] = EXPECT["nanochat_total"] - EXPECT["apertus_overlap_unique"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--selected", required=True)
    p.add_argument("--drop-list", required=True)
    # Key columns are REQUIRED and explicit (codex BLOCKER-1/2): no silent auto-pick.
    p.add_argument("--selected-key-col", required=True, help="Doc-key column in SELECTED.")
    p.add_argument("--drop-key-col", required=True, help="Key column in the drop-list (same key-space).")
    p.add_argument("--source-col", required=True, help="Source column in SELECTED (e.g. source_dataset).")
    p.add_argument("--output-manifest", required=True)
    p.add_argument("--batch-size", type=int, default=1_000_000)
    p.add_argument("--expect-drop-keys", type=int, default=EXPECT["apertus_overlap_unique"],
                   help="Expected DISTINCT drop-list keys; mismatch → REVIEW.")
    p.add_argument("--expect-selected-rows-upper", type=int, default=EXPECT["selected_rows_upper_bound"])
    p.add_argument("--expect-selected-rows-lower", type=int, default=80_000_000,
                   help="Lower bound on SELECTED rows; a truncated file below this → REVIEW (codex BLOCKER-2).")
    p.add_argument("--check-dup-keys", action="store_true",
                   help="Also compute distinct SELECTED keys (memory ~ keys*60B; needs a big node).")
    p.add_argument("--allow-review-exit-zero", action="store_true",
                   help="Exit 0 even on REVIEW verdict (default: exit 2 so automation never ships a bad artifact).")
    a = p.parse_args()
    if a.batch_size <= 0 or a.batch_size > 50_000_000:
        sys.exit(f"ERROR: --batch-size out of range (got {a.batch_size})")
    return a


def _dataset(path: str) -> pads.Dataset:
    # Handles a single parquet file OR a partitioned directory; validates a
    # single unified schema across fragments (codex MAJOR-7).
    return pads.dataset(path, format="parquet")


def _require_col(ds: pads.Dataset, col: str, where: str) -> pa.DataType:
    if col not in ds.schema.names:
        sys.exit(f"ERROR: column '{col}' not in {where} schema {ds.schema.names}")
    return ds.schema.field(col).type


def load_drop_keys(path: str, col: str) -> tuple[pa.Array, dict]:
    ds = _dataset(path)
    typ = _require_col(ds, col, "drop-list")
    rows = nulls = empties = 0
    seen: set = set()
    for batch in ds.to_batches(columns=[col], batch_size=1_000_000):
        arr = batch[col]
        rows += len(arr)
        nulls += pc.sum(pc.is_null(arr)).as_py() or 0
        if pa.types.is_string(typ) or pa.types.is_large_string(typ):
            empties += pc.sum(pc.equal(pc.fill_null(arr, "x"), "")).as_py() or 0
        for v in arr.to_pylist():
            if v is not None and v != "":
                seen.add(v)
    import hashlib
    h = hashlib.sha256()
    for f in sorted(getattr(ds, "files", []) or []):
        try:
            with open(f, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
        except OSError:
            pass
    stats = {"rows": rows, "nulls": nulls, "empties": empties, "distinct": len(seen),
             "type": str(typ), "sha256": h.hexdigest()}
    return pa.array(sorted(seen), type=typ), stats   # sorted → deterministic (codex MINOR-12)


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=os.path.dirname(os.path.abspath(__file__)),
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def file_meta(path: str) -> dict:
    try:
        st = os.stat(path)
        m = {"size_bytes": st.st_size, "mtime": int(st.st_mtime)}
        try:
            m["parquet_num_rows_footer"] = pq.ParquetFile(path).metadata.num_rows
        except Exception:
            pass
        return m
    except OSError:
        return {}


def atomic_write_json(path: str, obj: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2)
        fh.flush(); os.fsync(fh.fileno())
    os.replace(tmp, path)


def main() -> None:
    a = parse_args()
    t0 = time.time()

    drop_arr, drop_stats = load_drop_keys(a.drop_list, a.drop_key_col)
    print(f"[stage1] drop-list: {drop_stats}", flush=True)
    drop_ok = (drop_stats["rows"] == drop_stats["distinct"] == a.expect_drop_keys
               and drop_stats["nulls"] == 0 and drop_stats["empties"] == 0)  # exact (codex BLOCKER-3)
    if not drop_ok:
        print(f"[stage1][WARN] drop-list integrity off: distinct={drop_stats['distinct']} "
              f"expected={a.expect_drop_keys} nulls={drop_stats['nulls']} empties={drop_stats['empties']}",
              flush=True)

    sds = _dataset(a.selected)
    sel_key_t = _require_col(sds, a.selected_key_col, "SELECTED")
    src_t = _require_col(sds, a.source_col, "SELECTED")
    same_keyspace = (str(sel_key_t) == drop_stats["type"])  # exact Arrow type identity (codex BLOCKER-2/MAJOR-9)
    if not same_keyspace:
        print(f"[stage1][WARN] key-space mismatch: SELECTED key type {sel_key_t} != "
              f"drop-list key type {drop_stats['type']}; overlap check is NOT sound.", flush=True)

    # Footer row count: sum across all fragments (handles single file OR dir) — codex MAJOR-10.
    sel_meta_rows = 0
    for f in (getattr(sds, "files", []) or []):
        try:
            sel_meta_rows += pq.ParquetFile(f).metadata.num_rows
        except Exception:
            sel_meta_rows = None; break

    per_source: collections.Counter = collections.Counter()
    null_source = 0
    rows = in_droplist = null_keys = empty_keys = 0
    distinct_keys: set | None = set() if a.check_dup_keys else None

    drop_values = drop_arr if isinstance(drop_arr, pa.Array) else drop_arr.combine_chunks()  # build ONCE (codex MAJOR-8)
    for batch in sds.to_batches(columns=[a.selected_key_col, a.source_col], batch_size=a.batch_size):
        keys = batch[a.selected_key_col]
        srcs = batch[a.source_col]
        rows += len(keys)
        # null/empty keys (vectorized)
        null_keys += pc.sum(pc.is_null(keys)).as_py() or 0
        if pa.types.is_string(sel_key_t) or pa.types.is_large_string(sel_key_t):
            empty_keys += pc.sum(pc.equal(pc.fill_null(keys, "x"), "")).as_py() or 0
        # membership in drop-list (vectorized is_in) — only counts non-null
        if same_keyspace:
            isin = pc.is_in(keys, value_set=drop_values)
            in_droplist += pc.sum(pc.if_else(pc.is_null(isin), False, isin)).as_py() or 0
        # per-source value counts (vectorized), merge
        vc = pc.value_counts(srcs)
        for s in vc:
            val = s["values"].as_py()
            if val is None:
                null_source += s["counts"].as_py()
            else:
                per_source[val] += s["counts"].as_py()
        if distinct_keys is not None:
            for k in keys.to_pylist():
                if k is not None and k != "":
                    distinct_keys.add(k)
        if rows % (a.batch_size * 10) == 0:
            print(f"[stage1] scanned {rows} rows…", flush=True)

    drop_overlap_zero = (in_droplist == 0)
    apertus_overlap_drop_applied = bool(drop_overlap_zero and same_keyspace and drop_ok)
    rows_ok = (sel_meta_rows is None or rows == sel_meta_rows)
    within_bounds = a.expect_selected_rows_lower <= rows <= a.expect_selected_rows_upper
    implied_internal_dedup_drops = a.expect_selected_rows_upper - rows
    hplt_in_sel = per_source.get("HPLT/ell_Grek_ge8_no_mt_clean60", 0)
    hplt_ok = 40_000_000 <= hplt_in_sel <= EXPECT["nanochat_hplt_clean60"]
    sources_ok = (null_source == 0)
    dup_clean = (distinct_keys is not None and (rows - len(distinct_keys) - null_keys - empty_keys) == 0)

    # Tri-level verdict (codex BLOCKER-2/MAJOR-5/6): PASS requires the internal-dedup
    # distinct-key check to have run; otherwise the apertus-drop + counts are verified
    # but internal-dedup distinctness is only producer-attested.
    base_ok = (apertus_overlap_drop_applied and rows_ok and within_bounds and sources_ok
               and hplt_ok and null_keys == 0 and empty_keys == 0 and drop_ok)
    verdict = "PASS" if (base_ok and dup_clean) else ("PASS_APERTUS_ONLY" if base_ok else "REVIEW")

    manifest = {
        "artifact": "deduped_dataset",
        "verdict": verdict,
        "selected_parquet": a.selected,
        "drop_list_parquet": a.drop_list,
        "rows_scanned": rows,
        "rows_footer_metadata": sel_meta_rows,
        "rows_match_footer": rows_ok,
        "selected_rows_upper_bound_expected": a.expect_selected_rows_upper,
        "selected_rows_lower_bound_expected": a.expect_selected_rows_lower,
        "rows_within_bounds": within_bounds,
        "implied_internal_dedup_drops": implied_internal_dedup_drops,
        "keys": {
            "selected_key_col": a.selected_key_col, "selected_key_type": str(sel_key_t),
            "drop_key_col": a.drop_key_col, "drop_key_type": drop_stats["type"],
            "same_keyspace": same_keyspace,
        },
        "drop_list_integrity": {**drop_stats, "expected_distinct": a.expect_drop_keys, "ok": drop_ok},
        "apertus_overlap": {
            "selected_keys_in_droplist": in_droplist,
            "drop_overlap_zero": drop_overlap_zero,
            "apertus_overlap_drop_applied": apertus_overlap_drop_applied,
        },
        "null_keys": null_keys, "empty_keys": empty_keys, "null_source_rows": null_source,
        "distinct_keys": (len(distinct_keys) if distinct_keys is not None else None),
        "duplicate_key_occurrences": (rows - len(distinct_keys) - null_keys - empty_keys
                                      if distinct_keys is not None else None),
        "per_source_dataset": dict(sorted(per_source.items(), key=lambda x: -x[1])),
        "source_sanity": {
            "hplt_clean60_in_selected": hplt_in_sel,
            "hplt_expected_after_apertus_drop_approx": EXPECT["nanochat_hplt_clean60"] - 2_099_717,
            "hplt_ok": hplt_ok, "sources_ok": sources_ok, "null_source_rows": null_source,
            "n_distinct_sources": len(per_source),
        },
        "expected_nanochat_totals": EXPECT,
        "provenance": {
            "text_source": "fffoivos/glossapi-greek-nanochat-pretraining-dataset",
            "apertus_overlap_audit": "fffoivos/apertus-c3-dedup-audit-dedup-20260519t010924z",
            "internal_dedup_bundle": "wave2_20260426_builder_metadata_v2_latest_cleaner_20260507",
            "selected_file_meta": file_meta(a.selected),
            "drop_list_file_meta": file_meta(a.drop_list),
            "git_commit": git_commit(), "pyarrow_version": pa.__version__,
            "hostname": socket.gethostname(), "argv": sys.argv,
        },
        "internal_dedup_note": ("distinct-key check ran" if a.check_dup_keys else
                                "internal dedup is producer-attested (nanochat bundle); "
                                "re-verify with --check-dup-keys"),
        "wall_seconds": round(time.time() - t0, 1),
    }
    atomic_write_json(a.output_manifest, manifest)
    print(f"[stage1] verdict={verdict}  drop_applied={apertus_overlap_drop_applied} "
          f"(in_droplist={in_droplist})  rows={rows} (footer={sel_meta_rows}, ≤bound={within_bounds})  "
          f"sources={len(per_source)}  null_keys={null_keys}", flush=True)
    print(f"[stage1] manifest → {a.output_manifest}", flush=True)
    # Non-zero exit on REVIEW so automation never treats a failed validation as
    # success (codex BLOCKER-1). PASS / PASS_APERTUS_ONLY exit 0.
    if verdict == "REVIEW" and not a.allow_review_exit_zero:
        sys.exit(2)


if __name__ == "__main__":
    main()
