#!/usr/bin/env python3
"""One streaming pass over the SELECTED CPT parquet → per-source {id,text} JSONL.

The production academic text feeding CPT lives in
`selected_after_apertus_and_internal_dedup.parquet` (doc-level `text`, `source_dataset`,
`source_doc_id`). This splits the academic rows into one JSONL per source in a single
columnar streaming pass (constant memory), so the Rust detector can run per source.

NOTE on section sources: in the SELECTED parquet, Kallipos/Pergamos are doc-level
(sections already concatenated → the `β` label is gone), so they are detected in
whole-doc mode here, same as greek_phd/openarchives. The section-`β` lever only exists
on the raw section parquets (see ../driver/run_reference_detect.py --mode sections).
"""
import argparse
import json
import os
import re
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selected", required=True, help="selected_after_apertus_and_internal_dedup.parquet")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--id-col", default="source_doc_id")
    ap.add_argument("--source-col", default="source_dataset")
    # source -> regex on source_dataset
    ap.add_argument("--map", action="append", default=[
        "greek_phd=^greek_phd",
        "openarchives=^openarchives\\.gr",
        "kallipos=^Apothetirio_Kallipos",
        "pergamos=^Apothetirio_Pergamos",
    ], help="repeatable name=regex")
    a = ap.parse_args()

    import pyarrow.parquet as pq

    os.makedirs(a.out_dir, exist_ok=True)
    pats = {}
    for m in a.map:
        name, rx = m.split("=", 1)
        pats[name] = re.compile(rx)
    writers = {name: open(os.path.join(a.out_dir, f"{name}.jsonl"), "w", encoding="utf-8") for name in pats}
    counts = {name: 0 for name in pats}

    pf = pq.ParquetFile(a.selected)
    cols = [f.name for f in pf.schema_arrow]
    need = [c for c in [a.id_col, a.text_col, a.source_col] if c in cols]
    missing = [c for c in [a.id_col, a.text_col, a.source_col] if c not in cols]
    if missing:
        sys.exit(f"missing columns {missing} in {a.selected}; have {cols}")

    n = 0
    for b in pf.iter_batches(batch_size=20000, columns=need):
        d = b.to_pydict()
        ids = d[a.id_col]
        txt = d[a.text_col]
        src = d[a.source_col]
        for i in range(len(ids)):
            s = src[i] or ""
            for name, rx in pats.items():
                if rx.search(s):
                    writers[name].write(json.dumps({"id": ids[i], "text": txt[i] or ""}, ensure_ascii=False) + "\n")
                    counts[name] += 1
                    break
        n += len(ids)
        if n % 2_000_000 == 0:
            print(f"  scanned {n:,} rows; per-source so far: {counts}", file=sys.stderr, flush=True)
    for w in writers.values():
        w.close()
    print(f"[split] scanned {n:,} rows → {a.out_dir}; per-source: {counts}", file=sys.stderr)


if __name__ == "__main__":
    main()
