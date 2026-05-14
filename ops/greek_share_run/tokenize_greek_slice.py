"""Tokenize the Greek slice of each Apertus-pretraining dataset with the Apertus tokenizer.

One subcommand per Greek-bearing dataset. Each writes outputs/<dataset>.json with
exact (doc, utf8_bytes, tokens) counts. BOD/EOD special tokens are added per doc
to match how Apertus counts pretraining tokens.

Datasets handled:
  fineweb2_hq_ell        — epfml/FineWeb2-HQ config ell_Grek (60 parquet, ~83 GB)
  clean_wikipedia_el     — HuggingFaceFW/clean-wikipedia path el/ (3 parquet, ~0.6 GB)
  europarl_el            — Helsinki-NLP/europarl, 20 Greek-containing bitexts
  paradocs_el            — jhu-clsp/paradocs Greek-bearing pairs
  euroblocks_el          — utter-project/EuroBlocks-SFT-Synthetic-1124 filtered language=='el'
  institutional_books_el — institutional/institutional-books-1.0 filtered Greek-bearing
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Iterable, Iterator

import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from tokenizers import Tokenizer


APERTUS_REPO = "swiss-ai/Apertus-8B-2509"
BOD_EOD_PER_DOC = 2  # <s> + </s> per doc, paper §2.1
LOG_EVERY_DOCS = 250_000
BATCH = 10_000


def log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def get_tokenizer() -> Tokenizer:
    log(f"loading tokenizer from {APERTUS_REPO}")
    return Tokenizer.from_pretrained(APERTUS_REPO)


def iter_parquet_text(paths: list[Path], text_field: str) -> Iterator[str]:
    """Stream `text_field` values from a list of parquet files."""
    for p in paths:
        pf = pq.ParquetFile(p)
        for batch in pf.iter_batches(batch_size=20_000, columns=[text_field]):
            for val in batch.column(text_field).to_pylist():
                if val:
                    yield val


def encode_count(texts: Iterable[str], tok: Tokenizer, label: str) -> tuple[int, int, int]:
    n_docs = n_bytes = n_tokens = 0
    buf: list[str] = []
    t0 = time.time()
    last_log = 0

    def flush() -> None:
        nonlocal n_tokens, buf
        if not buf:
            return
        # add_special_tokens=False so we count content tokens only; the +2 BOD/EOD
        # per doc is added analytically at the end (matches Apertus paper §2.1).
        encs = tok.encode_batch(buf, add_special_tokens=False)
        n_tokens += sum(len(e.ids) for e in encs)
        buf = []

    for text in texts:
        if not text:
            continue
        n_docs += 1
        n_bytes += len(text.encode("utf-8"))
        buf.append(text)
        if len(buf) >= BATCH:
            flush()
        if n_docs - last_log >= LOG_EVERY_DOCS:
            dt = time.time() - t0
            log(
                f"  {label}: {n_docs:,} docs, "
                f"{n_bytes/1e9:.2f} GB UTF-8, "
                f"{n_tokens/1e9:.3f} B tokens, "
                f"{dt:.0f}s "
                f"({n_docs/max(dt,1):,.0f} docs/s, "
                f"{n_bytes/max(dt,1)/1e6:,.1f} MB/s)"
            )
            last_log = n_docs

    flush()
    # add BOD+EOD per doc to match Apertus's training-token convention
    n_tokens_with_specials = n_tokens + n_docs * BOD_EOD_PER_DOC
    return n_docs, n_bytes, n_tokens_with_specials


def download_dataset_files(
    repo_id: str,
    path_in_repo: str | None,
    dest_dir: Path,
    allow_patterns: list[str] | None = None,
    max_workers: int = 16,
) -> list[Path]:
    """Download parquet files under path_in_repo to dest_dir in parallel, return local paths.

    Uses snapshot_download with max_workers for parallelism and built-in retry; this
    avoids the CLOSE-WAIT hang we saw with serial hf_hub_download on large slices.
    """
    api = HfApi()
    siblings = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    pattern_prefix = (path_in_repo.rstrip("/") + "/") if path_in_repo else ""
    parquet_under = [
        f for f in siblings if f.startswith(pattern_prefix) and f.endswith(".parquet")
    ]
    if allow_patterns:
        parquet_under = [
            f for f in parquet_under if any(p in f for p in allow_patterns)
        ]
    log(
        f"downloading {len(parquet_under)} parquet files from {repo_id} "
        f"under {path_in_repo!r} (parallel max_workers={max_workers})"
    )
    dest_dir.mkdir(parents=True, exist_ok=True)
    if not parquet_under:
        return []
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(dest_dir),
        allow_patterns=[pattern_prefix + "*.parquet"] if pattern_prefix else ["**/*.parquet"],
        max_workers=max_workers,
    )
    local_paths = sorted(
        [dest_dir / f for f in parquet_under if (dest_dir / f).exists()]
    )
    log(f"  downloaded {len(local_paths)}/{len(parquet_under)} files")
    return local_paths


# ============================================================================
# Dataset-specific drivers
# ============================================================================


def driver_fineweb2_hq_ell(tok: Tokenizer, scratch: Path) -> dict:
    repo = "epfml/FineWeb2-HQ"
    dst = scratch / "fineweb2_hq_ell"
    paths = download_dataset_files(repo, "ell_Grek", dst)
    n_docs, n_bytes, n_tokens = encode_count(
        iter_parquet_text(paths, "text"), tok, "fineweb2_hq_ell"
    )
    return {
        "dataset": repo,
        "selector": "config=ell_Grek (path ell_Grek/)",
        "text_field": "text",
        "parquet_files": len(paths),
        "parquet_bytes_local": sum(p.stat().st_size for p in paths),
        "docs": n_docs,
        "utf8_bytes": n_bytes,
        "tokens_with_bod_eod": n_tokens,
        "tokens_text_only": n_tokens - n_docs * BOD_EOD_PER_DOC,
        "notes": (
            "Tokens include +2 BOD/EOD per doc to match Apertus pretrain accounting. "
            "Apertus consumed a subset of this slice per stage: × p × 0.95 sampler "
            "(p=0.33 in Stages 1-3, p=0.10 in Stages 4-5)."
        ),
    }


def driver_clean_wikipedia_el(tok: Tokenizer, scratch: Path) -> dict:
    repo = "HuggingFaceFW/clean-wikipedia"
    dst = scratch / "clean_wikipedia_el"
    paths = download_dataset_files(repo, "el", dst)
    n_docs, n_bytes, n_tokens = encode_count(
        iter_parquet_text(paths, "text"), tok, "clean_wikipedia_el"
    )
    return {
        "dataset": repo,
        "selector": "path el/",
        "text_field": "text",
        "parquet_files": len(paths),
        "parquet_bytes_local": sum(p.stat().st_size for p in paths),
        "docs": n_docs,
        "utf8_bytes": n_bytes,
        "tokens_with_bod_eod": n_tokens,
        "tokens_text_only": n_tokens - n_docs * BOD_EOD_PER_DOC,
        "notes": "Used only in Stage 5 (33 B-token Clean-Wikipedia slice in Table 6).",
    }


def driver_europarl_el(tok: Tokenizer, scratch: Path) -> dict:
    """EuroParl Greek side. We tokenize the `el` column of every Greek-containing bitext.

    Apertus's `europarl/main_bidirectional.py` consumes each bitext as a separate
    document, so Greek text shows up once per Greek-bearing pair (20 pairs). We
    record per-pair counts and a total that reflects per-pair multiplicity.
    """
    repo = "Helsinki-NLP/europarl"
    api = HfApi()
    info = api.dataset_info(repo)
    all_pairs = []
    for c in info.card_data.get("configs", []):
        name = c.get("config_name") if isinstance(c, dict) else c
        if name and ("el-" in name or "-el" in name) and len(name) == 5:
            all_pairs.append(name)
    log(f"europarl Greek pairs: {all_pairs} ({len(all_pairs)})")

    per_pair = {}
    total_docs = total_bytes = total_tokens = 0
    dst = scratch / "europarl_el"
    for pair in sorted(all_pairs):
        log(f"  pulling pair {pair}")
        # determine which side is 'el'
        el_side = "el"
        # Helsinki-NLP/europarl uses translation feature with two keys named after the pair codes
        # We download parquet then read the translation column
        # The file layout uses subdirs per pair config
        pair_dst = dst / pair
        # Most HF parquet datasets store under `{config}/{split}-*.parquet`
        api2 = HfApi()
        siblings = api2.list_repo_files(repo, repo_type="dataset")
        pair_files = [
            f for f in siblings if f.startswith(pair + "/") and f.endswith(".parquet")
        ]
        if not pair_files:
            log(f"    no parquet files for pair {pair} — skipping")
            continue
        local_paths = []
        for f in pair_files:
            lp = hf_hub_download(repo, f, repo_type="dataset", local_dir=pair_dst)
            local_paths.append(Path(lp))

        # extract Greek side
        def el_texts():
            for p in local_paths:
                pf = pq.ParquetFile(p)
                # columns are usually `translation` (struct) — peek to confirm
                schema = pf.schema_arrow
                cols = [f.name for f in schema]
                if "translation" in cols:
                    for batch in pf.iter_batches(batch_size=20_000, columns=["translation"]):
                        for rec in batch.column("translation").to_pylist():
                            if rec and isinstance(rec, dict) and rec.get(el_side):
                                yield rec[el_side]
                else:
                    # fall back: if column 'el' exists directly
                    if el_side in cols:
                        for batch in pf.iter_batches(batch_size=20_000, columns=[el_side]):
                            for v in batch.column(el_side).to_pylist():
                                if v:
                                    yield v

        nd, nb, nt = encode_count(el_texts(), tok, f"europarl[{pair}]")
        per_pair[pair] = {"docs": nd, "utf8_bytes": nb, "tokens_with_bod_eod": nt}
        total_docs += nd
        total_bytes += nb
        total_tokens += nt
        # release parquet files for this pair to keep disk usage bounded
        for p in local_paths:
            try:
                p.unlink()
            except FileNotFoundError:
                pass

    return {
        "dataset": repo,
        "selector": f"Greek side of bitexts: {sorted(all_pairs)}",
        "per_pair": per_pair,
        "docs": total_docs,
        "utf8_bytes": total_bytes,
        "tokens_with_bod_eod": total_tokens,
        "tokens_text_only": total_tokens - total_docs * BOD_EOD_PER_DOC,
        "notes": (
            "Total reflects Greek text appearing once per Greek-bearing pair "
            "(20 pairs). Apertus's europarl/main_bidirectional.py consumes each pair "
            "as a separate doc, so this multiplicity matches consumption."
        ),
    }


def driver_paradocs_el(tok: Tokenizer, scratch: Path) -> dict:
    """ParaDocs Greek pairs.

    ParaDocs lacks an explicit config-per-pair layout. We list repo files and grep
    for Greek-pair indicators in filenames; then stream and tokenize the Greek side.
    """
    repo = "jhu-clsp/paradocs"
    api = HfApi()
    siblings = api.list_repo_files(repo, repo_type="dataset")
    candidates = [
        f for f in siblings
        if f.endswith(".parquet") and ("ell" in f.lower() or "_el_" in f.lower() or "/el/" in f.lower() or "-el-" in f.lower() or "el." in f.lower())
    ]
    log(f"paradocs Greek candidate files: {len(candidates)}; first 5: {candidates[:5]}")

    dst = scratch / "paradocs_el"
    n_docs = n_bytes = n_tokens = 0
    per_file = {}
    for f in candidates:
        lp = hf_hub_download(repo, f, repo_type="dataset", local_dir=dst)
        path = Path(lp)
        pf = pq.ParquetFile(path)
        schema_names = [fl.name for fl in pf.schema_arrow]
        # ParaDocs uses 'translation' struct keyed by ISO codes; or per-pair columns
        def texts():
            for batch in pf.iter_batches(batch_size=20_000):
                d = batch.to_pylist()
                for rec in d:
                    # try common keys
                    for key in ("el", "ell", "ell_Grek"):
                        v = rec.get(key) if isinstance(rec, dict) else None
                        if v:
                            yield v
                            break
                    else:
                        t = rec.get("translation") if isinstance(rec, dict) else None
                        if isinstance(t, dict):
                            for k in ("el", "ell"):
                                if t.get(k):
                                    yield t[k]
                                    break
        nd, nb, nt = encode_count(texts(), tok, f"paradocs[{f}]")
        per_file[f] = {"docs": nd, "utf8_bytes": nb, "tokens_with_bod_eod": nt, "schema": schema_names}
        n_docs += nd; n_bytes += nb; n_tokens += nt
        try:
            path.unlink()
        except FileNotFoundError:
            pass

    return {
        "dataset": repo,
        "selector": "filename heuristic for Greek-bearing pairs",
        "per_file": per_file,
        "docs": n_docs,
        "utf8_bytes": n_bytes,
        "tokens_with_bod_eod": n_tokens,
        "tokens_text_only": n_tokens - n_docs * BOD_EOD_PER_DOC,
        "notes": "ParaDocs file layout heuristic — verify candidates list before trusting.",
    }


def driver_euroblocks_el(tok: Tokenizer, scratch: Path) -> dict:
    repo = "utter-project/EuroBlocks-SFT-Synthetic-1124"
    api = HfApi()
    siblings = api.list_repo_files(repo, repo_type="dataset")
    parquet_files = [f for f in siblings if f.endswith(".parquet")]
    dst = scratch / "euroblocks_el"
    log(f"euroblocks parquet files: {len(parquet_files)}")

    n_docs = n_bytes = n_tokens = 0
    for f in parquet_files:
        lp = hf_hub_download(repo, f, repo_type="dataset", local_dir=dst)
        path = Path(lp)
        pf = pq.ParquetFile(path)
        cols = [c.name for c in pf.schema_arrow]
        # Filter rows where language == 'el'
        # We need to read messages content
        def el_texts():
            # EuroBlocks: prefer `language` (declared target language during synthesis) over
            # `langid` (fastText-derived; misclassifies Greek conversations as English when
            # the prompt scaffolding is English). Verified 2026-05-11: language=='Greek' has
            # 582 rows, langid=='el' has 1.
            lang_col = "language" if "language" in cols else ("langid" if "langid" in cols else None)
            msg_col = (
                "conversations" if "conversations" in cols
                else ("messages" if "messages" in cols else ("text" if "text" in cols else None))
            )
            if not lang_col or not msg_col:
                return
            for batch in pf.iter_batches(batch_size=20_000, columns=[lang_col, msg_col]):
                langs = batch.column(lang_col).to_pylist()
                msgs = batch.column(msg_col).to_pylist()
                for lang, m in zip(langs, msgs):
                    if (lang or "").lower() not in ("el", "ell", "greek", "modern greek"):
                        continue
                    if isinstance(m, list):
                        parts = []
                        for turn in m:
                            if isinstance(turn, dict):
                                role = turn.get("from") or turn.get("role") or ""
                                content = turn.get("value") or turn.get("content") or ""
                                parts.append(f"{role}: {content}")
                            else:
                                parts.append(str(turn))
                        yield "\n".join(parts)
                    elif m:
                        yield str(m)

        nd, nb, nt = encode_count(el_texts(), tok, f"euroblocks[{f}]")
        n_docs += nd; n_bytes += nb; n_tokens += nt
        try:
            path.unlink()
        except FileNotFoundError:
            pass

    return {
        "dataset": repo,
        "selector": "language == 'Greek' (declared target language)",
        "docs": n_docs,
        "utf8_bytes": n_bytes,
        "tokens_with_bod_eod": n_tokens,
        "tokens_text_only": n_tokens - n_docs * BOD_EOD_PER_DOC,
        "notes": "Rendered as 'role: content\\n' per turn for tokenization.",
    }


def driver_institutional_books_el(tok: Tokenizer, scratch: Path) -> dict:
    """Greek volumes in Institutional Books 1.0.

    Each volume has a `language_distribution_gen` field with ISO 693-3 codes.
    Greek = `ell`. We retain a volume's text if `ell` is among its languages.
    """
    repo = "institutional/institutional-books-1.0"
    api = HfApi()
    siblings = api.list_repo_files(repo, repo_type="dataset")
    parquet_files = [f for f in siblings if f.endswith(".parquet")]
    log(f"institutional_books parquet files: {len(parquet_files)}")
    dst = scratch / "institutional_books_el"

    n_docs = n_bytes = n_tokens = 0
    for i, f in enumerate(parquet_files):
        lp = hf_hub_download(repo, f, repo_type="dataset", local_dir=dst)
        path = Path(lp)
        pf = pq.ParquetFile(path)
        cols = [c.name for c in pf.schema_arrow]

        text_col = None
        for c in ("text", "content", "body"):
            if c in cols:
                text_col = c; break
        lang_col = None
        for c in ("language_distribution_gen", "language", "languages"):
            if c in cols:
                lang_col = c; break
        if not text_col or not lang_col:
            log(f"  WARN missing fields in {f}: cols={cols}")
            try: path.unlink()
            except FileNotFoundError: pass
            continue

        def el_texts():
            for batch in pf.iter_batches(batch_size=2_000, columns=[lang_col, text_col]):
                langs = batch.column(lang_col).to_pylist()
                txts = batch.column(text_col).to_pylist()
                for lang_field, t in zip(langs, txts):
                    if not t:
                        continue
                    is_greek = False
                    if isinstance(lang_field, dict):
                        keys = lang_field.get("languages") or lang_field.get("language") or []
                        if isinstance(keys, list) and "ell" in keys:
                            is_greek = True
                    elif isinstance(lang_field, list):
                        if "ell" in lang_field:
                            is_greek = True
                    elif isinstance(lang_field, str):
                        if lang_field == "ell":
                            is_greek = True
                    if is_greek:
                        yield t

        nd, nb, nt = encode_count(el_texts(), tok, f"inst_books[{i+1}/{len(parquet_files)}]")
        n_docs += nd; n_bytes += nb; n_tokens += nt
        try: path.unlink()
        except FileNotFoundError: pass
        gc.collect()

    return {
        "dataset": repo,
        "selector": "language_distribution_gen contains 'ell'",
        "docs": n_docs,
        "utf8_bytes": n_bytes,
        "tokens_with_bod_eod": n_tokens,
        "tokens_text_only": n_tokens - n_docs * BOD_EOD_PER_DOC,
        "notes": "Long-context phase only (paper §3.4, 28.7B-token mixture).",
    }


DRIVERS = {
    "fineweb2_hq_ell": driver_fineweb2_hq_ell,
    "clean_wikipedia_el": driver_clean_wikipedia_el,
    "europarl_el": driver_europarl_el,
    "paradocs_el": driver_paradocs_el,
    "euroblocks_el": driver_euroblocks_el,
    "institutional_books_el": driver_institutional_books_el,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=sorted(DRIVERS.keys()))
    ap.add_argument("--scratch", default="/mnt/data/scratch")
    ap.add_argument("--out", default="/mnt/data/outputs")
    args = ap.parse_args()

    scratch = Path(args.scratch)
    out_dir = Path(args.out)
    scratch.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    log(f"=== driver: {args.dataset} ===")
    log(f"scratch: {scratch}")
    log(f"out: {out_dir}")

    tok = get_tokenizer()
    t0 = time.time()
    result = DRIVERS[args.dataset](tok, scratch)
    result["wall_seconds"] = round(time.time() - t0, 2)
    result["dataset_key"] = args.dataset

    out_path = out_dir / f"{args.dataset}.json"
    with out_path.open("w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    log(f"wrote {out_path} (wall {result['wall_seconds']}s)")


if __name__ == "__main__":
    main()
