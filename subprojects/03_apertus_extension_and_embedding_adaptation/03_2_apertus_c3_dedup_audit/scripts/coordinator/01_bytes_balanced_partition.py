#!/usr/bin/env python3
"""Coordinator step 01: enumerate input shards and bin-pack into 8 worker configs.

**Scope clarification (per review 2026-05-18 round 4)**:

This audit is the **`hf_source_pool` mode** of the dedup audit. It measures
overlap between Apertus pretraining sources and the *broader HF source pool*
from which the C3 BPE-training mix was sampled (`fffoivos/glossapi-greek-nanochat-pretraining-dataset`
+ `fffoivos/hplt-greek-ge8-no-mt-clean60-wave4`). It does NOT measure the
exact 1:1 sampled C3 mix/train view — that would require the C3 mix manifest
which lives on the (currently TERMINATED) `apertus-greek-tokenizer-20260408t160000z`
instance. The `c3_exact_mix_overlap` mode is a separate future audit step;
this run produces `hf_source_pool_overlap` results.

All audit outputs are named/labelled `hf_source_pool_*`, NOT `c3_*`, to keep
the scope unambiguous downstream.

Inputs:
- text_dedup_pin.json source list
- HF API (via `huggingface_hub.HfApi().repo_info(files_metadata=True)`)

Outputs:
- manifests/run_<RUN_ID>/partition.json
- manifests/run_<RUN_ID>/worker_<0..7>.json (per-worker configs)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from huggingface_hub import HfApi

SUB = Path(__file__).resolve().parents[2]
RUN_ID = (SUB / "manifests/CURRENT_RUN_ID").read_text().strip()
MANI = SUB / f"manifests/run_{RUN_ID}"
PIN = json.loads((MANI / "text_dedup_pin.json").read_text())

# 20 EuroParl Greek bitexts (Helsinki-NLP/europarl): every `el-X` or `X-el` config.
EUROPARL_EL_BITEXTS = [
    "bg-el", "cs-el", "da-el", "de-el", "el-en", "el-es", "el-et", "el-fi",
    "el-fr", "el-hu", "el-it", "el-lt", "el-lv", "el-nl", "el-pl", "el-pt",
    "el-ro", "el-sk", "el-sl", "el-sv",
]


def expand_sources() -> list[dict]:
    """Per-source spec including explicit-file filter (no glob substring matching)."""
    api = HfApi()
    out: list[dict] = []

    # --- Apertus side ---
    # 1. FW2-HQ ell_Grek — `ell_Grek/*.parquet` config dir.
    out.append({
        "corpus_id": "apertus", "source_id": "fw2hq_ell_grek",
        "hf_repo": "epfml/FineWeb2-HQ", "hf_repo_type": "dataset",
        "hf_config": "ell_Grek",
        "file_filter": lambda fn: fn.startswith("ell_Grek/") and fn.endswith(".parquet"),
    })

    # 2. Clean-Wikipedia el — `el/*.parquet` config dir.
    out.append({
        "corpus_id": "apertus", "source_id": "cleanwiki_el",
        "hf_repo": "HuggingFaceFW/clean-wikipedia", "hf_repo_type": "dataset",
        "hf_config": "el",
        "file_filter": lambda fn: fn.startswith("el/") and fn.endswith(".parquet"),
    })

    # 3. EuroParl Greek — 20 explicit bitexts. Use a precise prefix list, NOT a
    # substring glob (the substring approach would over-match `aeleal-*` etc.).
    out.append({
        "corpus_id": "apertus", "source_id": "europarl_greek",
        "hf_repo": "Helsinki-NLP/europarl", "hf_repo_type": "dataset",
        "hf_config": None,
        "file_filter": lambda fn: (
            fn.endswith(".parquet")
            and any(fn.startswith(b + "/") for b in EUROPARL_EL_BITEXTS)
        ),
    })

    # 4. EuroBlocks Greek — pull both parquet shards; worker filters
    # `language == 'Greek'` row-level.
    out.append({
        "corpus_id": "apertus", "source_id": "euroblocks_greek",
        "hf_repo": "utter-project/EuroBlocks-SFT-Synthetic-1124",
        "hf_repo_type": "dataset", "hf_config": None,
        "file_filter": lambda fn: fn.startswith("data/") and fn.endswith(".parquet"),
    })

    # --- HF source pool side (was misnamed `c3` in v1; corrected per review r4) ---
    # 5. GlossAPI nanochat release.
    out.append({
        "corpus_id": "hf_source_pool", "source_id": "glossapi_nanochat",
        "hf_repo": "fffoivos/glossapi-greek-nanochat-pretraining-dataset",
        "hf_repo_type": "dataset", "hf_config": None,
        "file_filter": lambda fn: fn.startswith("data/") and fn.endswith(".parquet"),
    })

    # 6. HPLT clean60 Greek release.
    out.append({
        "corpus_id": "hf_source_pool", "source_id": "hplt_clean60",
        "hf_repo": "fffoivos/hplt-greek-ge8-no-mt-clean60-wave4",
        "hf_repo_type": "dataset", "hf_config": None,
        "file_filter": lambda fn: fn.startswith("data/") and fn.endswith(".parquet"),
    })

    for entry in out:
        info = api.repo_info(repo_id=entry["hf_repo"],
                              repo_type=entry["hf_repo_type"],
                              files_metadata=True)
        files = []
        for sib in info.siblings:
            fn = sib.rfilename
            if entry["file_filter"](fn):
                files.append({"file": fn, "bytes_est": int(sib.size or 0)})
        entry["files_with_sizes"] = files
        entry["total_bytes"] = sum(f["bytes_est"] for f in files)
        del entry["file_filter"]  # not JSON serialisable
        print(f"[partition] {entry['source_id']}: {len(files)} files, "
              f"{entry['total_bytes'] / 1e9:.2f} GB")
    return out


def main() -> int:
    sources = expand_sources()
    # Flatten into shard list.
    all_shards: list[dict] = []
    for src in sources:
        for f in src["files_with_sizes"]:
            all_shards.append({
                "corpus_id": src["corpus_id"],
                "source_id": src["source_id"],
                "hf_repo": src["hf_repo"],
                "hf_repo_type": src["hf_repo_type"],
                "hf_config": src.get("hf_config"),
                "file": f["file"],
                "bytes_est": f["bytes_est"],
            })

    # Greedy bin-pack into N workers by bytes. N defaults to 8, override via env.
    import os
    N_WORKERS = int(os.environ.get("WORKER_COUNT", "8"))
    all_shards.sort(key=lambda s: -s["bytes_est"])
    workers: list[list[dict]] = [[] for _ in range(N_WORKERS)]
    bytes_per_worker = [0] * N_WORKERS
    for sh in all_shards:
        i = min(range(N_WORKERS), key=lambda j: bytes_per_worker[j])
        workers[i].append(sh)
        bytes_per_worker[i] += sh["bytes_est"]

    partition = {
        "run_id": RUN_ID,
        "scope": "hf_source_pool",  # NOT c3_exact_mix
        "total_shards": len(all_shards),
        "total_bytes_est": sum(bytes_per_worker),
        "per_worker_bytes_est": bytes_per_worker,
        "shards": all_shards,
    }
    (MANI / "partition.json").write_text(json.dumps(partition, ensure_ascii=False, indent=2))

    bucket = PIN["bucket"]
    for i, ws in enumerate(workers):
        by_source: dict[str, dict] = {}
        for sh in ws:
            key = sh["source_id"]
            if key not in by_source:
                by_source[key] = {
                    "corpus_id": sh["corpus_id"],
                    "source_id": sh["source_id"],
                    "hf_repo": sh["hf_repo"],
                    "hf_repo_type": sh["hf_repo_type"],
                    "hf_config": sh.get("hf_config"),
                    "files": [],
                    "bytes_est": 0,
                }
            by_source[key]["files"].append(sh["file"])
            by_source[key]["bytes_est"] += sh["bytes_est"]
        cfg = {
            "worker_idx": i,
            "run_id": RUN_ID,
            "bucket": bucket,
            "scope": "hf_source_pool",
            "shards": list(by_source.values()),
        }
        (MANI / f"worker_{i}.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2))
        print(f"[partition] worker_{i}: {len(ws)} shards, "
              f"{bytes_per_worker[i] / 1e9:.2f} GB")
    print(f"[partition] wrote partition.json + 8 worker_<i>.json under {MANI}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
