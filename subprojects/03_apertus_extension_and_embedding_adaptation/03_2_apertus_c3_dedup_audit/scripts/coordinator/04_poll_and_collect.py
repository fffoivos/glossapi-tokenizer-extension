#!/usr/bin/env python3
"""Coordinator step 04: supervisor loop + post-completion validation.

Polls each worker every 60s for stage sentinels (via GCS bucket polling, no
ssh per cycle), surfaces progress in a single line per minute. If any worker
uploads `_failed`, exits quickly. When all N_WORKERS _done sentinels are
uploaded, runs validation:

  (a) Pull each worker's run_state/{pull_log.jsonl,hash_log.jsonl} from GCS
      and check for any *_error / *_failed entries. ANY error = fail.
  (b) Sum the expected shard count from per-worker config files
      (manifests/run_<RUN_ID>/worker_<i>.json) and compare to actually
      uploaded shard count in GCS. Anything less than expected = fail.

Outputs: manifests/run_<RUN_ID>/progress.jsonl (one event per poll cycle)
         manifests/run_<RUN_ID>/validation.json (post-completion summary)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

SUB = Path(__file__).resolve().parents[2]
RUN_ID = (SUB / "manifests/CURRENT_RUN_ID").read_text().strip()
MANI = SUB / f"manifests/run_{RUN_ID}"
PIN = json.loads((MANI / "text_dedup_pin.json").read_text())
BUCKET = PIN["bucket"]
_workers_file = MANI / "workers.list"
N_WORKERS = sum(1 for ln in _workers_file.read_text().splitlines() if ln.strip()) \
            if _workers_file.exists() else 8
POLL_INTERVAL_S = 60
DEADLINE_S = int(os.environ.get("POLL_DEADLINE_S", str(60 * 60 * 6)))

# Failure markers seen in pull_log.jsonl / hash_log.jsonl.
ERROR_EVENTS = {"file_pull_error", "snapshot_pull_error", "shard_failed"}


def gcs_ls(path: str) -> list[str]:
    r = subprocess.run(
        ["gcloud", "storage", "ls", path],
        capture_output=True, text=True,
    )
    return [ln.strip() for ln in r.stdout.splitlines() if ln.strip()]


def gcs_cat(path: str) -> str | None:
    r = subprocess.run(
        ["gcloud", "storage", "cat", path],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        return None
    return r.stdout


def expected_shard_count_per_worker() -> dict[int, int]:
    """For each worker, count `files` entries in its worker_<i>.json — one
    parquet input → one shard output. Matches the per-shard output naming in
    hash_pass.py (one output file per input file)."""
    counts: dict[int, int] = {}
    for i in range(N_WORKERS):
        cfg_path = MANI / f"worker_{i}.json"
        if not cfg_path.exists():
            counts[i] = 0
            continue
        cfg = json.loads(cfg_path.read_text())
        n = 0
        for shard in cfg.get("shards", []):
            n += len(shard.get("files", []))
        counts[i] = n
    return counts


def validate_completion() -> dict:
    """Run post-_done validation. Returns dict with overall pass/fail + per-worker detail."""
    expected = expected_shard_count_per_worker()
    report = {"per_worker": {}, "overall_pass": True, "failure_reasons": []}
    for i in range(N_WORKERS):
        w = {"expected_input_shards": expected[i],
             "expected_uploaded_shards": None,
             "uploaded_shards": 0,
             "processed_shards": 0,
             "zero_row_shards": 0,
             "pull_errors": 0, "hash_errors": 0,
             "pass": True, "reasons": []}
        # Count uploaded shard parquets (exclude sentinel + run_state).
        uploaded = gcs_ls(f"{BUCKET}/worker_{i}/")
        shard_parquets = [u for u in uploaded if u.endswith(".parquet")]
        w["uploaded_shards"] = len(shard_parquets)
        # Pull pull_log.jsonl.
        pull_content = gcs_cat(f"{BUCKET}/worker_{i}/run_state/pull_log.jsonl")
        if pull_content is None:
            w["pass"] = False
            w["reasons"].append("missing pull_log.jsonl")
        else:
            for line in pull_content.splitlines():
                try:
                    rec = json.loads(line.strip())
                except Exception:
                    continue
                if rec.get("event") in ERROR_EVENTS:
                    w["pull_errors"] += 1
            if w["pull_errors"] > 0:
                w["pass"] = False
                w["reasons"].append(f"{w['pull_errors']} pull_errors")

        # Pull hash_log.jsonl. Some input shards legitimately produce zero
        # retained rows after source-specific Greek filtering, so uploaded
        # parquet count must equal positive-row shard count, not raw input count.
        hash_content = gcs_cat(f"{BUCKET}/worker_{i}/run_state/hash_log.jsonl")
        if hash_content is None:
            w["pass"] = False
            w["reasons"].append("missing hash_log.jsonl")
        else:
            expected_uploaded = 0
            for line in hash_content.splitlines():
                try:
                    rec = json.loads(line.strip())
                except Exception:
                    continue
                ev = rec.get("event")
                if ev in ERROR_EVENTS:
                    w["hash_errors"] += 1
                if ev in {"shard_done", "shard_skipped_existing"}:
                    w["processed_shards"] += 1
                    rows_kept = int(rec.get("rows_kept") or 0)
                    if rows_kept > 0:
                        expected_uploaded += 1
                    else:
                        w["zero_row_shards"] += 1
            w["expected_uploaded_shards"] = expected_uploaded
            if w["hash_errors"] > 0:
                w["pass"] = False
                w["reasons"].append(f"{w['hash_errors']} hash_errors")
            if w["processed_shards"] < w["expected_input_shards"]:
                w["pass"] = False
                w["reasons"].append(
                    f"processed {w['processed_shards']} < expected inputs {w['expected_input_shards']}"
                )
            if w["uploaded_shards"] != expected_uploaded:
                w["pass"] = False
                w["reasons"].append(
                    f"uploaded {w['uploaded_shards']} != expected positive-row outputs {expected_uploaded}"
                )
        report["per_worker"][f"worker_{i}"] = w
        if not w["pass"]:
            report["overall_pass"] = False
            report["failure_reasons"].append(f"worker_{i}: {'; '.join(w['reasons'])}")
    return report


def main() -> int:
    log = (MANI / "progress.jsonl").open("a")
    start = time.time()
    seen_done = set()
    seen_failed = set()
    while True:
        elapsed = time.time() - start
        if elapsed > DEADLINE_S:
            print(f"[poll] DEADLINE exceeded ({DEADLINE_S}s); halting", file=sys.stderr)
            log.write(json.dumps({"event": "deadline_exceeded", "elapsed_s": int(elapsed)}) + "\n")
            return 3
        # Check each worker's _done sentinel.
        for i in range(N_WORKERS):
            if i in seen_done:
                continue
            failed_path = f"{BUCKET}/worker_{i}/_failed"
            failed_listing = gcs_ls(failed_path)
            if failed_listing:
                seen_failed.add(i)
                failure_detail = gcs_cat(failed_path)
                print(f"[poll] worker {i} FAILED at {time.strftime('%H:%M:%S')} "
                      f"(elapsed {int(elapsed)}s): {failure_detail}", file=sys.stderr)
                log.write(json.dumps({"event": "worker_failed", "worker_idx": i,
                                       "elapsed_s": int(elapsed),
                                       "detail": failure_detail}) + "\n")
                log.flush()
                (MANI / "validation.json").write_text(json.dumps({
                    "overall_pass": False,
                    "failure_reasons": [f"worker_{i} uploaded _failed"],
                    "failed_worker": i,
                    "failed_detail": failure_detail,
                }, ensure_ascii=False, indent=2))
                return 4
            done_path = f"{BUCKET}/worker_{i}/_done"
            listing = gcs_ls(done_path)
            if listing:
                seen_done.add(i)
                print(f"[poll] worker {i} DONE at {time.strftime('%H:%M:%S')} "
                      f"(elapsed {int(elapsed)}s)")
                log.write(json.dumps({"event": "worker_done", "worker_idx": i,
                                       "elapsed_s": int(elapsed)}) + "\n")
                log.flush()
        # Live progress: count shard outputs per worker.
        progress = []
        for i in range(N_WORKERS):
            shard_count = len(gcs_ls(f"{BUCKET}/worker_{i}/"))
            progress.append(shard_count)
        progress_str = " ".join(f"w{i}={n}" for i, n in enumerate(progress))
        print(f"[poll] t+{int(elapsed)}s  done={len(seen_done)}/{N_WORKERS}  shards: {progress_str}")
        log.write(json.dumps({"event": "tick", "elapsed_s": int(elapsed),
                               "done_count": len(seen_done),
                               "shards_per_worker": progress}) + "\n")
        log.flush()
        if len(seen_done) == N_WORKERS:
            print(f"[poll] all {N_WORKERS} workers DONE; running validation ...")
            log.write(json.dumps({"event": "all_done", "elapsed_s": int(elapsed)}) + "\n")
            log.flush()
            break
        time.sleep(POLL_INTERVAL_S)

    # Post-completion validation.
    report = validate_completion()
    (MANI / "validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"[poll] validation: overall_pass={report['overall_pass']}")
    for wid, w in report["per_worker"].items():
        print(f"  {wid}: pass={w['pass']}  "
              f"processed={w['processed_shards']}/{w['expected_input_shards']}  "
              f"uploaded={w['uploaded_shards']}/{w['expected_uploaded_shards']}  "
              f"zero_row_shards={w['zero_row_shards']}  "
              f"pull_errors={w['pull_errors']}  hash_errors={w['hash_errors']}")
    if not report["overall_pass"]:
        print("[poll] FATAL — validation FAILED:", file=sys.stderr)
        for r in report["failure_reasons"]:
            print(f"  - {r}", file=sys.stderr)
        log.write(json.dumps({"event": "validation_failed", "reasons": report["failure_reasons"]}) + "\n")
        return 4
    log.write(json.dumps({"event": "validation_passed"}) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
