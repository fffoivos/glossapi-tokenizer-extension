"""Stage A: build a coverage-balanced Greek document set for the added-token audit.

Greedy supply-first selection: stream the CPT source parquets, tokenise with the
extended tokenizer, and keep a document only if it supplies an added token that
is still below the per-token floor K.  Guarantees the rare tail is sampled
instead of drowned by the head, which a uniform random sample would not do.
Tokenisation is checkpoint-independent, so this runs once for all checkpoints.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--plan", required=True, help="json list of {file, register, max_docs}")
    p.add_argument("--out-jsonl", required=True)
    p.add_argument("--out-coverage", required=True)
    p.add_argument("--new-id-range", nargs=2, type=int, default=[131072, 148992])
    p.add_argument("--floor", type=int, default=16)
    p.add_argument("--max-chars", type=int, default=6000)
    p.add_argument("--min-chars", type=int, default=400)
    p.add_argument("--max-keep", type=int, default=60000)
    p.add_argument("--batch", type=int, default=512)
    return p.parse_args()



def _batches(path, kind, batch, pq):
    """Yield lists of raw texts from a jsonl or parquet source."""
    if kind == "jsonl":
        buf = []
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = row.get("text")
                if text:
                    buf.append(text)
                if len(buf) >= batch:
                    yield buf
                    buf = []
        if buf:
            yield buf
        return
    pf = pq.ParquetFile(path)
    col = "text" if "text" in pf.schema_arrow.names else pf.schema_arrow.names[0]
    for rb in pf.iter_batches(batch_size=batch, columns=[col]):
        yield rb.column(0).to_pylist()


def main():
    a = parse_args()
    import numpy as np, pyarrow.parquet as pq
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    lo, hi = a.new_id_range
    n_new = hi - lo
    counts = np.zeros(n_new, dtype=np.int64)
    plan = json.loads(Path(a.plan).read_text())
    kept = 0
    t0 = time.time()
    out = open(a.out_jsonl, "w", encoding="utf-8")

    for entry in plan:
        path = Path(a.data_root) / entry["file"]
        if not path.exists():
            print("MISSING", path, flush=True)
            continue
        reg = entry["register"]
        budget = entry.get("max_docs", 10**9)
        seen = 0
        for texts_raw in _batches(path, entry.get("kind", "parquet"), a.batch, pq):
            texts = [t for t in texts_raw if t]
            texts = [t[: a.max_chars] for t in texts if len(t) >= a.min_chars]
            if not texts:
                continue
            enc = tok(texts, add_special_tokens=False)["input_ids"]
            for text, ids in zip(texts, enc):
                seen += 1
                arr = np.asarray(ids)
                new = arr[(arr >= lo) & (arr < hi)] - lo
                if new.size == 0:
                    continue
                uniq = np.unique(new)
                if not bool((counts[uniq] < a.floor).any()):
                    continue                       # supplies nothing still needed
                np.add.at(counts, new, 1)
                out.write(json.dumps({"text": text, "register": reg},
                                     ensure_ascii=False) + "\n")
                kept += 1
                if kept % 2000 == 0:
                    below = int((counts < a.floor).sum())
                    print(f"kept={kept} src={entry['file']} below_floor={below}/{n_new} "
                          f"{time.time()-t0:.0f}s", flush=True)
                if kept >= a.max_keep:
                    break
            if seen >= budget or kept >= a.max_keep:
                break
        print(f"done {entry['file']}: scanned={seen} kept_total={kept} "
              f"below_floor={int((counts < a.floor).sum())}", flush=True)
        if kept >= a.max_keep:
            break
        if int((counts < a.floor).sum()) == 0:
            print("floor reached for every added token", flush=True)
            break
    out.close()

    below = counts < a.floor
    cov = {
        "floor": a.floor, "n_docs_kept": kept, "n_added_tokens": int(n_new),
        "n_below_floor": int(below.sum()),
        "n_zero": int((counts == 0).sum()),
        "modern_below_floor": int(below[: 148480 - lo].sum()),
        "polytonic_below_floor": int(below[148480 - lo:].sum()),
        "count_percentiles": {p: float(np.percentile(counts, p))
                              for p in (0, 1, 5, 25, 50, 75, 95, 100)},
        "zero_token_ids": (np.where(counts == 0)[0] + lo).tolist()[:2000],
        "elapsed_s": time.time() - t0,
    }
    Path(a.out_coverage).write_text(json.dumps(cov, indent=1))
    print(json.dumps({k: v for k, v in cov.items() if k != "zero_token_ids"}, indent=1))


if __name__ == "__main__":
    main()
