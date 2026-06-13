#!/usr/bin/env python3
"""Thin I/O driver around the `reference_detect` Rust binary.

The driver does NO per-doc text work (no regex, no cleaning) — it only shapes I/O:
  - section sources (kallipos/pergamos): a SECTION-level parquet → group rows per
    document → newline-delimited {filename, rows:[...]} → pipe to `--mode sections`.
  - whole-doc sources (greek_phd/openarchives): the source `.jsonl.zst` is fed
    straight to `--mode wholedoc` (the Rust binary reads zstd + picks the text field).

All detection/segmentation/counting happens in Rust. Output: an auditable
`<source>/refspans/…` spans file + a per-doc `counters` file, plus an optional
parquet conversion of the counters for analysis. NOTHING is deleted; the keep/drop
policy is a separate, user-driven step over these emitted spans + counters.

Usage:
  # section source (parquet)
  run_reference_detect.py --source kallipos --mode sections \
      --input /…/Dataset_Kallipos.parquet --doc-col filename --out-dir out/kallipos [--limit N]
  # whole-doc source (jsonl.zst, possibly a glob)
  run_reference_detect.py --source greek_phd --mode wholedoc \
      --input '/…/greek_phd/.../contents/*.jsonl.zst' --out-dir out/greek_phd
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BIN = os.path.join(HERE, "..", "reference_detector", "target", "release", "reference_detect")


def build_sections_jsonl(parquet_path, doc_col, out_jsonl, limit=None):
    """Group a section-level parquet by document and emit one JSON line per doc.

    STREAMING + constant-memory: section parquets are filename-contiguous (verified
    for Kallipos/Pergamos — a filename's rows never interleave), so we iterate batches
    and flush each document as soon as its filename changes, never materialising the
    2.3 GB table. positional_fraction is computed per doc from row order. The only python
    work is assembling the grouped JSON envelope — pure I/O shaping, no per-row text work.
    A non-contiguous filename reappearance is detected and aborts loudly (use a pre-sort).
    """
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(parquet_path)
    cols = [f.name for f in pf.schema_arrow]
    need = [c for c in [doc_col, "id", "row_id", "header", "section", "predicted_section"] if c in cols]

    seen = set()
    cur_fn = None
    cur_rows = []  # list of (header, section, predicted_section, row_id)
    ndocs = 0
    nrows = 0

    def flush(w, fn, rows):
        m = max(1, len(rows) - 1)
        env = {
            "filename": fn,
            "rows": [
                {
                    "row_id": r[3] or f"row_{i}",
                    "header": r[0] or "",
                    "section": r[1] or "",
                    "predicted_section": r[2] or "",
                    "positional_fraction": (i / m),
                }
                for i, r in enumerate(rows)
            ],
        }
        w.write(json.dumps(env, ensure_ascii=False) + "\n")

    with open(out_jsonl, "w", encoding="utf-8") as w:
        for b in pf.iter_batches(batch_size=20000, columns=need):
            d = b.to_pydict()
            fns = d[doc_col]
            hs = d.get("header", [None] * len(fns))
            ss = d.get("section", [None] * len(fns))
            ps = d.get("predicted_section", [None] * len(fns))
            rid = d.get("row_id", [None] * len(fns))
            for i in range(len(fns)):
                fn = fns[i]
                if fn != cur_fn:
                    if cur_fn is not None:
                        flush(w, cur_fn, cur_rows)
                        ndocs += 1
                        if cur_fn in seen:
                            raise SystemExit(f"non-contiguous filename {cur_fn!r}; pre-sort the parquet")
                        seen.add(cur_fn)
                    cur_fn = fn
                    cur_rows = []
                cur_rows.append((hs[i], ss[i], ps[i], rid[i]))
                nrows += 1
                if limit and nrows >= limit:
                    break
            if limit and nrows >= limit:
                break
        if cur_fn is not None:
            flush(w, cur_fn, cur_rows)
            ndocs += 1
    return ndocs


def counters_to_parquet(counters_jsonl, out_parquet):
    import pyarrow as pa
    import pyarrow.parquet as pq

    recs = [json.loads(l) for l in open(counters_jsonl, encoding="utf-8") if l.strip()]
    if not recs:
        return 0
    tbl = pa.Table.from_pylist(recs)
    pq.write_table(tbl, out_parquet)
    return len(recs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--mode", choices=["sections", "wholedoc"], required=True)
    ap.add_argument("--input", required=True, help="parquet path (sections) or jsonl.zst glob (wholedoc)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--doc-col", default="filename")
    ap.add_argument("--bin", default=DEFAULT_BIN)
    ap.add_argument("--limit", type=int, default=None)
    # pass-through calibration knobs (defaults in the Rust binary = emit-everything)
    ap.add_argument("--knob", action="append", default=[], help="extra binary flag, e.g. --knob=--bib-min-year-density=0.2")
    a = ap.parse_args()

    os.makedirs(os.path.join(a.out_dir, "refspans"), exist_ok=True)
    spans = os.path.join(a.out_dir, "refspans", f"{a.source}.spans.jsonl")
    counters = os.path.join(a.out_dir, f"{a.source}.counters.jsonl")
    cmd = [a.bin, "--mode", a.mode, "--source", a.source, "--out-spans", spans, "--out-counters", counters]
    for k in a.knob:
        cmd += k.split("=", 1) if k.startswith("--") and "=" in k else [k]

    if a.mode == "sections":
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as tf:
            tmp = tf.name
        nd = build_sections_jsonl(a.input, a.doc_col, tmp, a.limit)
        print(f"[driver] grouped {nd} docs → {tmp}", file=sys.stderr)
        cmd += ["--input", tmp]
        subprocess.run(cmd, check=True)
        os.unlink(tmp)
    else:
        files = sorted(glob.glob(a.input)) if any(c in a.input for c in "*?[") else [a.input]
        if not files:
            sys.exit(f"no input files matched {a.input!r}")
        # concatenate decode via the binary one file at a time, appending
        first = True
        for fp in files:
            part_spans = spans + (".part" if not first else "")
            part_counters = counters + (".part" if not first else "")
            c2 = list(cmd)
            c2[c2.index("--out-spans") + 1] = part_spans
            c2[c2.index("--out-counters") + 1] = part_counters
            c2 += ["--input", fp]
            subprocess.run(c2, check=True)
            if not first:
                for base, part in [(spans, part_spans), (counters, part_counters)]:
                    with open(base, "a", encoding="utf-8") as out, open(part, encoding="utf-8") as inp:
                        out.write(inp.read())
                    os.unlink(part)
            first = False

    np = counters_to_parquet(counters, os.path.join(a.out_dir, f"{a.source}.counters.parquet"))
    print(f"[driver] {a.source}: spans→{spans}  counters→{counters} ({np} docs)  parquet written", file=sys.stderr)


if __name__ == "__main__":
    main()
