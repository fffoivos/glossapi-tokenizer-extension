#!/usr/bin/env python3
"""Verify checkpoint and sidecar state for one Vanilla 5B checkpoint.

Run from home. This is read-only: it checks the Megatron checkpoint, sidecar
watcher state, sidecar manifest, sidecar Slurm jobs, and local adversarial
review files without submitting or cancelling anything.
"""

import argparse
import csv
import datetime as dt
import json
from pathlib import Path
import re
import subprocess
import sys


DEFAULT_RUN_TAG = "04_vanilla_goldfish_5b_20260528T112539Z"
DEFAULT_RUN_ROOT = "/capstor/scratch/cscs/fffoivos/runs/04_vanilla_cpt"
EXPECTED_KINDS = [
    "convert",
    "native_mcq",
    "greek_nlp",
    "heldout_greek_bpb",
    "retention",
    "code_bpb",
    "math_bpb",
]


REMOTE_PROGRAM = r"""
import csv
import datetime as dt
import json
import os
from pathlib import Path
import subprocess

run_tag = os.environ["RUN_TAG"]
run_root = Path(os.environ["RUN_ROOT"])
iteration = int(os.environ["ITERATION"])
expected_kinds = os.environ["EXPECTED_KINDS"].split(",")
iter_pad = "%07d" % iteration

train_run_dir = run_root / run_tag
eval_root = run_root / ("eval_" + run_tag)
watch_dir = run_root / (run_tag + "_sidecar_watch")
checkpoint_dir = train_run_dir / "checkpoints" / ("iter_" + iter_pad)
iter_root = eval_root / ("iter_" + iter_pad)
hf_dir = eval_root / ("iter_" + iter_pad + "_hf")
manifest = iter_root / "sidecar_jobs.tsv"
submit_log = watch_dir / ("iter_%d_submit.log" % iteration)
submitted_state = watch_dir / ("iter_%d.submitted" % iteration)
checksum_dir = iter_root / "checksums"


def info(path):
    path = Path(path)
    item = {"path": str(path), "exists": path.exists()}
    if item["exists"]:
        stat = path.stat()
        item["size"] = stat.st_size
        item["mtime"] = dt.datetime.fromtimestamp(
            stat.st_mtime, dt.timezone.utc
        ).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        item["is_dir"] = path.is_dir()
        item["is_file"] = path.is_file()
        if item["is_dir"]:
            item["child_count"] = sum(1 for _ in path.iterdir())
    return item


def read_rows(path):
    if not Path(path).is_file():
        return []
    with Path(path).open() as f:
        return list(csv.DictReader(f, delimiter="\t"))


def run_cmd(args):
    proc = subprocess.run(
        args,
        universal_newlines=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return {
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def parse_squeue(ids):
    if not ids:
        return []
    result = run_cmd(
        ["squeue", "-h", "-j", ",".join(ids), "-o", "%i|%j|%T|%M|%l|%R"]
    )
    rows = []
    for line in result["stdout"].splitlines():
        parts = line.split("|", 5)
        if len(parts) != 6:
            continue
        rows.append(
            {
                "job_id": parts[0].strip(),
                "name": parts[1].strip(),
                "state": parts[2].strip(),
                "elapsed": parts[3].strip(),
                "limit": parts[4].strip(),
                "reason": parts[5].strip(),
            }
        )
    return rows


def parse_sacct(ids):
    if not ids:
        return []
    result = run_cmd(
        [
            "sacct",
            "-j",
            ",".join(ids),
            "--format=JobID,JobName%32,State,ExitCode,Elapsed,AllocTRES%80",
            "-P",
        ]
    )
    lines = [line for line in result["stdout"].splitlines() if line.strip()]
    if not lines:
        return []
    header = lines[0].split("|")
    return [dict(zip(header, line.split("|"))) for line in lines[1:]]


def ready_output(state):
    output = state["output"]
    if not output.get("exists"):
        return False
    if output.get("is_dir"):
        return output.get("child_count", 0) > 0
    return output.get("size", 0) > 0


manifest_rows = read_rows(manifest)
manifest_kinds = [row.get("kind", "") for row in manifest_rows]
job_ids = [row.get("job_id", "") for row in manifest_rows if row.get("job_id")]
missing_kinds = [kind for kind in expected_kinds if kind not in manifest_kinds]
output_states = [
    {
        "kind": row.get("kind", ""),
        "job_id": row.get("job_id", ""),
        "dependency": row.get("dependency", ""),
        "output": info(row.get("output", "")),
    }
    for row in manifest_rows
]
expected_output_states = [
    item for item in output_states if item["kind"] in expected_kinds
]
missing_outputs = [
    item["kind"] for item in expected_output_states if not ready_output(item)
]

checksum_candidates = sorted(
    checksum_dir.glob("*_iter_%s_checksum_manifest.json" % iter_pad)
)
checksum_manifest = (
    checksum_candidates[0]
    if checksum_candidates
    else checksum_dir / ("UNKNOWN_iter_%s_checksum_manifest.json" % iter_pad)
)

archived_attempts = []
for attempt_path in sorted(iter_root.glob("sidecar_jobs_attempt_*.tsv")):
    attempt_rows = read_rows(attempt_path)
    attempt_job_ids = [
        row.get("job_id", "") for row in attempt_rows if row.get("job_id")
    ]
    archived_attempts.append(
        {
            "file": info(attempt_path),
            "rows": attempt_rows,
            "job_ids": attempt_job_ids,
            "sacct": parse_sacct(attempt_job_ids),
        }
    )

squeue_rows = parse_squeue(job_ids)
sacct_rows = parse_sacct(job_ids)
primary_sacct_rows = [
    row for row in sacct_rows if row.get("JobID", "") in set(job_ids)
]
primary_sacct_by_id = {row.get("JobID", ""): row for row in primary_sacct_rows}
missing_sacct_job_ids = [
    job_id for job_id in job_ids if job_id not in primary_sacct_by_id
]
incomplete_sacct_jobs = [
    row
    for row in primary_sacct_rows
    if row.get("State") != "COMPLETED" or row.get("ExitCode") != "0:0"
]

state = {
    "checked_at_utc": dt.datetime.now(dt.timezone.utc)
    .replace(microsecond=0)
    .isoformat()
    .replace("+00:00", "Z"),
    "run_tag": run_tag,
    "iteration": iteration,
    "iter_pad": iter_pad,
    "expected_kinds": expected_kinds,
    "paths": {
        "train_run_dir": str(train_run_dir),
        "eval_root": str(eval_root),
        "watch_dir": str(watch_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "iter_eval_root": str(iter_root),
        "hf_dir": str(hf_dir),
        "manifest": str(manifest),
        "checksum_manifest": str(checksum_manifest),
    },
    "checkpoint": {
        "dir": info(checkpoint_dir),
        "metadata": info(checkpoint_dir / ".metadata"),
    },
    "watcher": {
        "submitted_state": info(submitted_state),
        "submit_log": info(submit_log),
    },
    "manifest": {
        "file": info(manifest),
        "rows": manifest_rows,
        "kinds": manifest_kinds,
        "missing_kinds": missing_kinds,
        "output_states": output_states,
        "missing_outputs": missing_outputs,
    },
    "slurm": {
        "job_ids": job_ids,
        "squeue": squeue_rows,
        "sacct": sacct_rows,
        "missing_sacct_job_ids": missing_sacct_job_ids,
        "incomplete_sacct_jobs": incomplete_sacct_jobs,
    },
    "checksum_manifest": info(checksum_manifest),
    "sidecar_attempt_history": {
        "archived_attempts": archived_attempts,
        "archived_attempt_count": len(archived_attempts),
        "archived_attempt_job_ids": [
            job_id
            for attempt in archived_attempts
            for job_id in attempt["job_ids"]
        ],
    },
}
manifest_has_all_expected_kinds = bool(manifest_rows) and not missing_kinds
expected_outputs_ready = manifest_has_all_expected_kinds and not missing_outputs
active_sidecar_jobs = len(state["slurm"]["squeue"])
slurm_jobs_completed = bool(job_ids) and not missing_sacct_job_ids and not incomplete_sacct_jobs
checksum_manifest_ready = (
    state["checksum_manifest"]["exists"]
    and state["checksum_manifest"].get("is_file", False)
    and state["checksum_manifest"].get("size", 0) > 0
)
state["summary"] = {
    "checkpoint_metadata_ready": state["checkpoint"]["metadata"]["exists"],
    "sidecars_submitted": state["watcher"]["submitted_state"]["exists"]
    and state["manifest"]["file"]["exists"],
    "manifest_has_all_expected_kinds": manifest_has_all_expected_kinds,
    "expected_outputs_ready": expected_outputs_ready,
    "active_sidecar_jobs": active_sidecar_jobs,
    "slurm_jobs_completed": slurm_jobs_completed,
    "checksum_manifest_ready": checksum_manifest_ready,
    "handoff_ready": (
        state["checkpoint"]["metadata"]["exists"]
        and state["watcher"]["submitted_state"]["exists"]
        and state["manifest"]["file"]["exists"]
        and manifest_has_all_expected_kinds
        and expected_outputs_ready
        and active_sidecar_jobs == 0
        and slurm_jobs_completed
        and checksum_manifest_ready
    ),
    "archived_sidecar_attempt_count": len(archived_attempts),
}
print(json.dumps(state, ensure_ascii=False, sort_keys=True))
"""


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--run-tag", default=DEFAULT_RUN_TAG)
    parser.add_argument("--run-root", default=DEFAULT_RUN_ROOT)
    parser.add_argument("--remote-host", default="clariden")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--compact", action="store_true")
    return parser.parse_args()


def run_remote(args):
    remote_env = (
        "RUN_TAG={run_tag!r} RUN_ROOT={run_root!r} ITERATION={iteration!r} "
        "EXPECTED_KINDS={kinds!r}"
    ).format(
        run_tag=args.run_tag,
        run_root=args.run_root,
        iteration=str(args.iteration),
        kinds=",".join(EXPECTED_KINDS),
    )
    cmd = [
        "ssh",
        "-o",
        "ConnectTimeout=30",
        args.remote_host,
        remote_env + " python3 - <<'PY'\n" + REMOTE_PROGRAM + "\nPY",
    ]
    proc = subprocess.run(
        cmd,
        universal_newlines=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        raise RuntimeError("remote verifier failed:\n{}".format(proc.stderr))
    return json.loads(proc.stdout)


def safe_label(label):
    return re.sub(r"(^_+|_+$)", "", re.sub(r"[^A-Za-z0-9._-]+", "_", label))


def file_info(path):
    path = Path(path)
    item = {"path": str(path), "exists": path.exists()}
    if item["exists"]:
        stat = path.stat()
        item["size"] = stat.st_size
        item["mtime"] = (
            dt.datetime.fromtimestamp(stat.st_mtime, dt.timezone.utc)
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z")
        )
    return item


def add_local_review_state(state):
    subproject_dir = Path(__file__).resolve().parents[1]
    labels = {
        119: "Vanilla-0.5B",
        238: "Vanilla-1B",
        477: "Vanilla-2B",
        834: "Vanilla-3.5B",
        1192: "Vanilla-5B",
    }
    label = labels.get(state["iteration"], "iter_%s" % state["iteration"])
    review_dir = subproject_dir / "adversarial_reviews" / safe_label(label)
    state["local_review"] = {
        "label": label,
        "dir": str(review_dir),
        "prompt": file_info(review_dir / "prompt.md"),
        "events_jsonl": file_info(review_dir / "codex_events.jsonl"),
        "metadata": file_info(review_dir / "review_metadata.env"),
        "critique": file_info(review_dir / "adversarial_critique.md"),
    }


def print_summary(state):
    summary = state["summary"]
    print("checked_at={}".format(state["checked_at_utc"]))
    print("iteration={}".format(state["iteration"]))
    print("checkpoint_metadata_ready={}".format(summary["checkpoint_metadata_ready"]))
    print("sidecars_submitted={}".format(summary["sidecars_submitted"]))
    print(
        "manifest_has_all_expected_kinds={}".format(
            summary["manifest_has_all_expected_kinds"]
        )
    )
    print("expected_outputs_ready={}".format(summary["expected_outputs_ready"]))
    print("active_sidecar_jobs={}".format(summary["active_sidecar_jobs"]))
    print("slurm_jobs_completed={}".format(summary["slurm_jobs_completed"]))
    print("checksum_manifest_ready={}".format(summary["checksum_manifest_ready"]))
    print("handoff_ready={}".format(summary["handoff_ready"]))
    print(
        "archived_sidecar_attempt_count={}".format(
            summary.get("archived_sidecar_attempt_count", 0)
        )
    )
    missing = state["manifest"]["missing_kinds"]
    if missing:
        print("missing_kinds={}".format(",".join(missing)))
    missing_outputs = state["manifest"].get("missing_outputs", [])
    if missing_outputs:
        print("missing_outputs={}".format(",".join(missing_outputs)))
    missing_sacct = state["slurm"].get("missing_sacct_job_ids", [])
    if missing_sacct:
        print("missing_sacct_job_ids={}".format(",".join(missing_sacct)))


def main():
    args = parse_args()
    state = run_remote(args)
    add_local_review_state(state)
    text = json.dumps(
        state,
        indent=None if args.compact else 2,
        ensure_ascii=False,
        sort_keys=True,
    ) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    print_summary(state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
