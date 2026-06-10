#!/usr/bin/env python3
"""Build the 5% Greek-replay source = nanochat docs whose (source_dataset,
source_doc_id) is in the apertus_overlap_drop list (the Apertus-original Greek
removed during dedup).

nanochat has no `doc_key`; the join key is the natural (source_dataset,
source_doc_id) pair. Streams the staged nanochat parquets, keeps matching rows,
writes (text, source_doc_id, source_dataset) to one parquet. CPU-only.
"""
import argparse, glob, os, sys
import pyarrow as pa
import pyarrow.parquet as pq

def load_drop_set(path):
    f = pq.ParquetFile(path)
    cols = f.schema_arrow.names
    assert "source_doc_id" in cols and "source_dataset" in cols, f"drop-list cols={cols}"
    s = set()
    for b in f.iter_batches(columns=["source_dataset", "source_doc_id"], batch_size=1_000_000):
        sd = b.column(0).to_pylist(); di = b.column(1).to_pylist()
        s.update(zip(sd, di))
    print(f"drop-list pairs: {len(s):,}", flush=True)
    return s

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drop-list", required=True)
    ap.add_argument("--nanochat-glob", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-chars", type=int, default=0)
    a = ap.parse_args()

    drop = load_drop_set(a.drop_list)
    files = sorted(glob.glob(a.nanochat_glob))
    print(f"nanochat files: {len(files)}", flush=True)
    os.makedirs(os.path.dirname(a.output), exist_ok=True)

    schema = pa.schema([("text", pa.string()), ("source_doc_id", pa.string()),
                        ("source_dataset", pa.string())])
    writer = pq.ParquetWriter(a.output, schema, compression="zstd")
    kept = chars = scanned = 0
    for fp in files:
        pf = pq.ParquetFile(fp)
        cols = pf.schema_arrow.names
        textcol = next((c for c in ("text", "content") if c in cols), None)
        if not textcol or "source_doc_id" not in cols or "source_dataset" not in cols:
            print(f"  skip {os.path.basename(fp)} (text={textcol})", flush=True); continue
        for b in pf.iter_batches(columns=["source_dataset", "source_doc_id", textcol], batch_size=20_000):
            d = b.to_pydict()
            sd, di, ts = d["source_dataset"], d["source_doc_id"], d[textcol]
            tk, ti, tsr = [], [], []
            for s, i, t in zip(sd, di, ts):
                scanned += 1
                if t and (s, i) in drop:
                    tk.append(t); ti.append(i); tsr.append(s); chars += len(t)
            if tk:
                writer.write_table(pa.table({"text": tk, "source_doc_id": ti, "source_dataset": tsr},
                                            schema=schema))
                kept += len(tk)
        print(f"  {os.path.basename(fp)}: kept={kept:,} chars={chars/1e9:.2f}B scanned={scanned/1e6:.1f}M",
              flush=True)
        if a.max_chars and chars >= a.max_chars:
            print("reached --max-chars, stopping", flush=True); break
    writer.close()
    print(f"DONE greek_replay: {kept:,} docs, ~{chars/1e9:.2f}B chars (~{chars/4/1e9:.2f}B tok est) -> {a.output}",
          flush=True)

if __name__ == "__main__":
    main()
