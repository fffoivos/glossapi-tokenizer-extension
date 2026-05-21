#!/usr/bin/env python3
"""Worker step: download assigned HF shards to /mnt/data/sources/<source_id>/.

Reads /mnt/data/run_state/worker_config.json with the structure produced by
coordinator/01_bytes_balanced_partition.py:

  {
    "worker_idx": 3,
    "shards": [
      {"corpus_id": "apertus", "source_id": "fw2hq_ell_grek",
       "hf_repo": "epfml/FineWeb2-HQ", "hf_config": "ell_Grek",
       "files": ["ell_Grek/train-00000-of-00060.parquet", ...],
       "bytes_est": 4823948213},
      ...
    ]
  }

For each shard entry, calls `huggingface_hub.hf_hub_download` (single-file form
when `files` is a small list) and stages files at /mnt/data/sources/<source_id>/.

Outputs:
- /mnt/data/run_state/pull_log.jsonl (one line per file with bytes + duration)
- /mnt/data/pull_done sentinel on success
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

CONFIG = Path("/mnt/data/run_state/worker_config.json")
LOG = Path("/mnt/data/run_state/pull_log.jsonl")
DONE = Path("/mnt/data/pull_done")
SRC_ROOT = Path("/mnt/data/sources")
HF_CACHE = Path(os.environ.get("HF_HOME", "/mnt/data/hf_cache"))
HF_TOKEN = os.environ.get("HF_TOKEN") or None


def log_event(event: dict) -> None:
    event["ts"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with LOG.open("a") as fh:
        fh.write(json.dumps(event, ensure_ascii=False) + "\n")


def pull_one_file(repo: str, filename: str, repo_type: str, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    path = hf_hub_download(
        repo_id=repo, filename=filename, repo_type=repo_type,
        cache_dir=str(HF_CACHE), token=HF_TOKEN, local_dir=str(dest_dir),
        local_dir_use_symlinks=False,
    )
    elapsed = time.time() - t0
    size = Path(path).stat().st_size
    log_event({"event": "file_pulled", "repo": repo, "file": filename,
               "bytes": size, "seconds": round(elapsed, 2)})
    return Path(path)


def main() -> int:
    from concurrent.futures import ThreadPoolExecutor, as_completed
    if not CONFIG.exists():
        print(f"[FATAL] missing config {CONFIG}", file=sys.stderr)
        return 2
    cfg = json.loads(CONFIG.read_text())
    widx = cfg["worker_idx"]
    log_event({"event": "pull_start", "worker_idx": widx,
               "shard_count": len(cfg["shards"])})
    # Build flat task list — one (repo, file, repo_type, dest) per file across all shards.
    tasks: list[tuple[str, str, str, Path]] = []
    snap_tasks: list[dict] = []
    for shard in cfg["shards"]:
        src_id = shard["source_id"]
        repo = shard["hf_repo"]
        repo_type = shard.get("hf_repo_type", "dataset")
        files = shard.get("files", [])
        dest = SRC_ROOT / src_id
        if not files:
            snap_tasks.append({"repo": repo, "repo_type": repo_type, "dest": dest,
                                "config": shard.get('hf_config', '')})
        else:
            for fn in files:
                tasks.append((repo, fn, repo_type, dest))

    # Per-file parallel pulls. 16 workers gives 16 × ~15 MB/s per stream ≈ 240 MB/s aggregate per worker.
    PULL_PARALLELISM = int(os.environ.get("PULL_PARALLELISM", "16"))
    n_pulled = 0
    failed: list[dict] = []
    with ThreadPoolExecutor(max_workers=PULL_PARALLELISM) as ex:
        futs = {ex.submit(pull_one_file, repo=repo, filename=fn, repo_type=repo_type,
                          dest_dir=dest): (repo, fn) for (repo, fn, repo_type, dest) in tasks}
        for f in as_completed(futs):
            try:
                f.result()
                n_pulled += 1
            except Exception as e:
                repo, fn = futs[f]
                rec = {"event": "file_pull_error", "repo": repo, "file": fn,
                       "error": repr(e)[:500]}
                log_event(rec)
                failed.append(rec)

    # Snapshot pulls (fallback path; rare).
    snap_failed: list[dict] = []
    for s in snap_tasks:
        t0 = time.time()
        try:
            snap = snapshot_download(
                repo_id=s["repo"], repo_type=s["repo_type"], cache_dir=str(HF_CACHE),
                token=HF_TOKEN, allow_patterns=[f"{s['config']}/**"],
                local_dir=str(s["dest"]), local_dir_use_symlinks=False,
            )
            log_event({"event": "snapshot_pulled", "repo": s["repo"], "snap": snap,
                        "seconds": round(time.time() - t0, 2)})
        except Exception as e:
            rec = {"event": "snapshot_pull_error", "repo": s["repo"],
                   "error": repr(e)[:500]}
            log_event(rec)
            snap_failed.append(rec)

    log_event({"event": "pull_done", "worker_idx": widx,
               "files_pulled": n_pulled, "files_failed": len(failed),
               "snapshots_failed": len(snap_failed),
               "tasks_total": len(tasks)})
    # CRITICAL: do NOT touch pull_done if anything failed. Upstream run_all has
    # set -e, so nonzero exit halts hash + upload + sentinel — worker won't
    # falsely report success.
    if failed or snap_failed:
        print(f"[pull] FATAL: {len(failed)} file pull errors + {len(snap_failed)} snapshot errors; see pull_log.jsonl. Not touching pull_done.", file=sys.stderr)
        return 3
    DONE.touch()
    return 0


if __name__ == "__main__":
    sys.exit(main())
