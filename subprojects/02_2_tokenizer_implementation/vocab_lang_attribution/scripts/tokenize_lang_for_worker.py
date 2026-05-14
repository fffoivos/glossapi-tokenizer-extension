"""Per-worker driver: tokenize the assigned canonical-language slice.

Reads pre-cached file listings from file_listings.json (avoid HF API rate limits).

For each canonical key in partition.json[WORKER_IDX]:
  1. From cached listing, find parquet shards for this language under each source.
  2. Shuffle shards (seed=42) and stream-download each via hf_hub_download.
  3. Tokenize with Apertus tokenizer (add_special_tokens=False).
  4. Accumulate per-canonical-key int64[131072] histogram via np.bincount.
  5. Stop when cumulative tokens for this key >= TOKEN_CAP.
  6. Write /mnt/data/outputs/arrays/<canonical_key>.npy and per-key summary JSON.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer


APERTUS_REPO = "swiss-ai/Apertus-8B-2509"
TOKEN_CAP_DEFAULT = 1_000_000_000
BATCH = 10_000
SHARD_SEED = 42
VOCAB_SIZE = 131_072


def log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def get_tokenizer() -> Tokenizer:
    return Tokenizer.from_pretrained(APERTUS_REPO)


def iter_text_from_parquet(path: Path, text_field: str = "text") -> Iterator[str]:
    try:
        pf = pq.ParquetFile(path)
    except Exception as e:
        log(f"  pq open failed: {path}: {e}")
        return
    cols = [f.name for f in pf.schema_arrow]
    if text_field not in cols:
        return
    for batch in pf.iter_batches(batch_size=20_000, columns=[text_field]):
        for v in batch.column(text_field).to_pylist():
            if v:
                yield v


def iter_text_from_parquet_translation_side(path: Path, side_code: str) -> Iterator[str]:
    """Extract one side of a translation pair (EuroParl / ParaDocs style)."""
    try:
        pf = pq.ParquetFile(path)
    except Exception as e:
        log(f"  pq open failed: {path}: {e}")
        return
    cols = [f.name for f in pf.schema_arrow]
    if "translation" not in cols:
        return
    for batch in pf.iter_batches(batch_size=20_000, columns=["translation"]):
        for rec in batch.column("translation").to_pylist():
            if rec and isinstance(rec, dict):
                v = rec.get(side_code)
                if v:
                    yield v


def tokenize_stream(text_iter, tok, histogram, token_cap, label, current_total):
    docs_added = 0; tokens_added = 0
    buf = []
    t0 = time.time(); last_log = 0

    def flush():
        nonlocal tokens_added, buf
        if not buf:
            return
        encs = tok.encode_batch(buf, add_special_tokens=False)
        ids = np.concatenate([np.asarray(e.ids, dtype=np.int64) for e in encs])
        np.add.at(histogram, ids, 1)
        tokens_added += int(ids.size)
        buf = []

    for text in text_iter:
        if not text:
            continue
        buf.append(text); docs_added += 1
        if len(buf) >= BATCH:
            flush()
            if current_total + tokens_added >= token_cap:
                break
        if docs_added - last_log >= 200_000:
            dt = time.time() - t0
            log(f"    {label}: {docs_added:,} docs, {tokens_added/1e9:.3f} B tokens, {dt:.0f}s")
            last_log = docs_added
    flush()
    return docs_added, tokens_added


def shuffled(lst, seed=SHARD_SEED):
    rng = random.Random(seed); out = list(lst); rng.shuffle(out); return out


def shards_for_source(file_listings: dict, repo: str, prefix: str) -> list[str]:
    """From cached listings, get parquet shards under `prefix`."""
    files = file_listings.get(repo, [])
    if not prefix:
        return [f for f in files if f.endswith(".parquet")]
    prefix_norm = prefix.rstrip("/") + "/"
    out = [f for f in files if f.startswith(prefix_norm) and f.endswith(".parquet")]
    if not out:
        # try also "data/<prefix>/" pattern (FW-2 sometimes uses this)
        alt = f"data/{prefix_norm}"
        out = [f for f in files if f.startswith(alt) and f.endswith(".parquet")]
    return out


def bitext_pair_shards(file_listings: dict, repo: str, side_code: str) -> tuple[str, list[str]]:
    """For EuroParl/ParaDocs: find one `en-XX` or `XX-en` config containing side_code."""
    files = file_listings.get(repo, [])
    # Try `en-<code>/` first, then `<code>-en/`, then any pair with the code
    for pair in (f"en-{side_code}", f"{side_code}-en"):
        shards = [f for f in files if f.startswith(pair + "/") and f.endswith(".parquet")]
        if shards:
            return pair, shards
    # any config containing the code
    for f in files:
        parts = f.split("/")
        if len(parts) > 1 and side_code in parts[0].split("-"):
            pair = parts[0]
            shards = [g for g in files if g.startswith(pair + "/") and g.endswith(".parquet")]
            return pair, shards
    return "", []


def process_one_canonical_key(canonical_key, canonical_meta, tok, scratch, token_cap, file_listings):
    histogram = np.zeros(VOCAB_SIZE, dtype=np.int64)
    cumulative = 0
    sources_used = []
    t0_key = time.time()
    sources = canonical_meta.get("sources", {})

    # Priority: fineweb_2_hq → fineweb_2 → clean_wikipedia → europarl → paradocs → fineweb_edu/hq/dclm
    source_specs = [
        ("fineweb_2_hq", "epfml/FineWeb2-HQ", sources.get("fineweb_2_hq"), "shard_text", "ell_Grek_style"),
        ("fineweb_2", "HuggingFaceFW/fineweb-2", sources.get("fineweb_2"), "shard_text", "fw2_style"),
        ("clean_wikipedia", "HuggingFaceFW/clean-wikipedia", sources.get("clean_wikipedia"), "shard_text", "wiki_list"),
        ("europarl", "Helsinki-NLP/europarl", sources.get("europarl"), "bitext_side", "ep_list"),
        ("paradocs", "jhu-clsp/paradocs", sources.get("paradocs"), "bitext_side", "pd_list"),
        ("fineweb_edu", "HuggingFaceFW/fineweb-edu", sources.get("fineweb_edu"), "shard_text", "single"),
        ("fineweb_hq", "epfml/FineWeb-HQ", sources.get("fineweb_hq"), "shard_text", "single"),
        ("dclm_edu", "HuggingFaceTB/dclm-edu", sources.get("dclm_edu"), "shard_text", "single"),
    ]

    for src_tag, repo, code, mode, layout in source_specs:
        if not code:
            continue
        if cumulative >= token_cap:
            break
        try:
            if mode == "shard_text":
                # Determine prefix(es) to scan
                if layout in ("ell_Grek_style", "fw2_style"):
                    prefixes = [code] if isinstance(code, str) else list(code)
                elif layout == "wiki_list":
                    prefixes = code if isinstance(code, list) else [code]
                else:  # single — no per-language prefix
                    prefixes = [""]
                pre = int(histogram.sum())
                docs = 0
                for prefix in prefixes:
                    if cumulative >= token_cap:
                        break
                    shards = shards_for_source(file_listings, repo, prefix)
                    if not shards:
                        log(f"  {canonical_key} {src_tag}: no shards for prefix={prefix!r}")
                        continue
                    shards = shuffled(shards)
                    for shard in shards:
                        if int(histogram.sum()) >= token_cap:
                            break
                        try:
                            lp = hf_hub_download(repo, shard, repo_type="dataset", local_dir=scratch / canonical_key / src_tag)
                        except Exception as e:
                            log(f"    {canonical_key} {src_tag}: download {shard} failed: {e}")
                            continue
                        d, _ = tokenize_stream(iter_text_from_parquet(Path(lp), "text"), tok, histogram,
                                               token_cap=token_cap, current_total=int(histogram.sum()),
                                               label=f"{canonical_key}/{src_tag}/{Path(shard).name}")
                        docs += d
                        try: Path(lp).unlink()
                        except FileNotFoundError: pass
                post = int(histogram.sum())
                sources_used.append({"tag": src_tag, "repo": repo, "tokens_added": post - pre, "docs_added": docs})
            elif mode == "bitext_side":
                # code is a list of 2-letter codes (eu/pd side codes)
                pre = int(histogram.sum())
                for side_code in (code if isinstance(code, list) else [code]):
                    if int(histogram.sum()) >= token_cap:
                        break
                    pair, shards = bitext_pair_shards(file_listings, repo, side_code)
                    if not shards:
                        continue
                    for shard in shards:
                        if int(histogram.sum()) >= token_cap:
                            break
                        try:
                            lp = hf_hub_download(repo, shard, repo_type="dataset", local_dir=scratch / canonical_key / f"{src_tag}_{pair}")
                        except Exception as e:
                            log(f"    {canonical_key} {src_tag}: download {shard} failed: {e}")
                            continue
                        d, _ = tokenize_stream(iter_text_from_parquet_translation_side(Path(lp), side_code),
                                               tok, histogram, token_cap=token_cap, current_total=int(histogram.sum()),
                                               label=f"{canonical_key}/{src_tag}/{pair}")
                        try: Path(lp).unlink()
                        except FileNotFoundError: pass
                post = int(histogram.sum())
                sources_used.append({"tag": src_tag, "repo": repo, "tokens_added": post - pre})
        except Exception as e:
            log(f"  {canonical_key} {src_tag}: source error {e}")
        cumulative = int(histogram.sum())

    summary = {
        "canonical_key": canonical_key,
        "iso_639_3": canonical_meta.get("iso_639_3"),
        "script": canonical_meta.get("script_iso15924"),
        "wall_seconds": round(time.time() - t0_key, 2),
        "sample_tokens_total": int(histogram.sum()),
        "vocab_entries_fired": int((histogram > 0).sum()),
        "vocab_entries_fired_geq_10": int((histogram >= 10).sum()),
        "sources_used": sources_used,
    }
    arr_dir = Path("/mnt/data/outputs/arrays"); arr_dir.mkdir(parents=True, exist_ok=True)
    np.save(arr_dir / f"{canonical_key}.npy", histogram)
    (arr_dir / f"{canonical_key}.summary.json").write_text(json.dumps(summary, ensure_ascii=False))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partition", required=True)
    ap.add_argument("--lang-map",  required=True)
    ap.add_argument("--file-listings", required=True)
    ap.add_argument("--worker-idx", type=int, required=True)
    ap.add_argument("--token-cap",  type=int, default=TOKEN_CAP_DEFAULT)
    ap.add_argument("--scratch",    default="/mnt/data/scratch")
    args = ap.parse_args()

    partition = json.load(open(args.partition))
    lang_map  = json.load(open(args.lang_map))
    file_listings = json.load(open(args.file_listings))
    my_keys   = partition[str(args.worker_idx)]
    log(f"worker {args.worker_idx}: {len(my_keys)} keys, file_listings has {len(file_listings)} repos")
    tok = get_tokenizer()
    scratch = Path(args.scratch); scratch.mkdir(parents=True, exist_ok=True)
    arr_dir = Path("/mnt/data/outputs/arrays"); arr_dir.mkdir(parents=True, exist_ok=True)

    for i, ckey in enumerate(my_keys, 1):
        if (arr_dir / f"{ckey}.npy").exists():
            continue  # resume — already done
        log(f"[{i}/{len(my_keys)}] {ckey}: starting")
        try:
            s = process_one_canonical_key(ckey, lang_map[ckey], tok, scratch, args.token_cap, file_listings)
            log(f"  -> {s['sample_tokens_total']:,} tokens, {s['vocab_entries_fired']:,} entries fired, {s['wall_seconds']:.0f}s")
        except Exception as e:
            log(f"  ERROR {ckey}: {e}")
            (arr_dir / f"{ckey}.error.txt").write_text(str(e))
        gc.collect()

    Path("/mnt/data/outputs/_done.flag").write_text(f"worker {args.worker_idx} done\n")
    log(f"worker {args.worker_idx} DONE")


if __name__ == "__main__":
    main()
