#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(message: str) -> None:
    print(f"[{utc_now()}] {message}", flush=True)


def run_cmd(cmd: list[str], *, input_text: str | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(cmd, input=input_text, text=True, capture_output=True)
    if check and proc.returncode != 0:
        raise RuntimeError(
            f"Command failed with code {proc.returncode}: {' '.join(shlex.quote(part) for part in cmd)}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )
    return proc


def remote_python(instance: str, project: str, zone: str, script: str) -> str:
    cmd = [
        "/home/foivos/google-cloud-sdk/bin/gcloud",
        "compute",
        "ssh",
        instance,
        "--project",
        project,
        "--zone",
        zone,
        "--command",
        "python3 -",
    ]
    return run_cmd(cmd, input_text=script).stdout


def remote_shell(instance: str, project: str, zone: str, command: str) -> subprocess.CompletedProcess[str]:
    cmd = [
        "/home/foivos/google-cloud-sdk/bin/gcloud",
        "compute",
        "ssh",
        instance,
        "--project",
        project,
        "--zone",
        zone,
        "--command",
        command,
    ]
    return run_cmd(cmd)


def build_status_script(c1_root: str, c2_root: str, c1_unit: str, c2_unit: str) -> str:
    return f"""
import glob, json, os, subprocess, time
runs = {{
  "c1": "{c1_root}",
  "c2": "{c2_root}",
}}
units = {{
  "c1": "{c1_unit}",
  "c2": "{c2_unit}",
}}
payload = {{"runs": {{}}, "units": {{}}}}
for name, unit in units.items():
    active = subprocess.run(["systemctl", "--user", "is-active", unit], capture_output=True, text=True)
    failed = subprocess.run(["systemctl", "--user", "is-failed", unit], capture_output=True, text=True)
    payload["units"][name] = {{
        "unit": unit,
        "active": active.stdout.strip() or active.stderr.strip(),
        "failed": failed.stdout.strip() or failed.stderr.strip(),
    }}
for name, root in runs.items():
    progress = os.path.join(root, "progress.json")
    summary = os.path.join(root, "training_summary.json")
    run_info = {{
        "root": root,
        "progress_exists": os.path.exists(progress),
        "summary_exists": os.path.exists(summary),
    }}
    if os.path.exists(progress):
        run_info["progress_mtime"] = int(os.stat(progress).st_mtime)
        with open(progress, "r", encoding="utf-8") as f:
            obj = json.load(f)
        for key in [
            "phase",
            "count_total_tasks",
            "count_completed_tasks",
            "count_inflight_tasks",
            "sequence_total_tasks",
            "sequence_completed_tasks",
            "sequence_inflight_tasks",
            "merge_target_added",
            "merge_completed_added",
            "current_vocab_size",
        ]:
            if key in obj:
                run_info[key] = obj[key]
    for sub in ["segment_shards", "sequence_shards"]:
        progress_dir = os.path.join(root, "work", sub, "_progress")
        files = sorted(glob.glob(os.path.join(progress_dir, "*.json")))
        if files:
            latest = max(files, key=lambda fp: os.stat(fp).st_mtime)
            run_info[f"{{sub}}_latest_mtime"] = int(os.stat(latest).st_mtime)
            with open(latest, "r", encoding="utf-8") as f:
                latest_obj = json.load(f)
            run_info[f"{{sub}}_latest_state"] = latest_obj.get("state")
    payload["runs"][name] = run_info
print(json.dumps(payload))
"""


def remote_status(args: argparse.Namespace) -> dict:
    output = remote_python(
        args.instance_name,
        args.project,
        args.zone,
        build_status_script(
            args.continuous_glossapi_only_run_dir,
            args.continuous_mix_run_dir,
            args.continuous_glossapi_only_unit,
            args.continuous_mix_unit,
        ),
    )
    return json.loads(output)


def status_line(status: dict) -> str:
    parts: list[str] = []
    for name in ["c1", "c2"]:
        run = status["runs"][name]
        unit = status["units"][name]
        parts.append(
            f"{name}: active={unit['active']} failed={unit['failed']} phase={run.get('phase')} "
            f"count={run.get('count_completed_tasks')}/{run.get('count_total_tasks')} "
            f"seq={run.get('sequence_completed_tasks')}/{run.get('sequence_total_tasks')} "
            f"merge={run.get('merge_completed_added')}/{run.get('merge_target_added')} "
            f"summary={run.get('summary_exists')}"
        )
    return " | ".join(parts)


def all_complete(status: dict) -> bool:
    return status["runs"]["c1"].get("summary_exists") and status["runs"]["c2"].get("summary_exists")


def any_failed(status: dict) -> bool:
    for name in ["c1", "c2"]:
        unit_state = status["units"][name]["failed"]
        if unit_state == "failed":
            return True
        if status["units"][name]["active"] == "inactive" and not status["runs"][name].get("summary_exists"):
            return True
    return False


def build_publish_command(args: argparse.Namespace) -> str:
    script_path = f"{args.worker_repo_root}/subprojects/02_1_tokenizer_experiments/scripts/publish_tokenizer_extension_repo.py"
    python_bin = args.worker_python
    env = [
        f"cd {shlex.quote(args.worker_repo_root)}",
        f"export PYTHONPATH={shlex.quote(args.worker_repo_root)}",
        f"{shlex.quote(python_bin)} {shlex.quote(script_path)} "
        f"--staging-dir {shlex.quote(args.publish_stage_dir)} "
        f"--repo-slug {shlex.quote(args.repo_slug)} "
        f"--fresh-run {shlex.quote('glossapi_only_50k=' + args.fresh_glossapi_only_run_dir)} "
        f"--fresh-run {shlex.quote('glossapi_plus_hplt_70_30_50k=' + args.fresh_mix_run_dir)} "
        f"--continuous-run {shlex.quote('continuous_glossapi_only_156672=' + args.continuous_glossapi_only_run_dir)} "
        f"--continuous-run {shlex.quote('continuous_glossapi_plus_hplt_70_30_156672=' + args.continuous_mix_run_dir)}"
    ]
    if args.repo_id:
        env[-1] += f" --repo-id {shlex.quote(args.repo_id)}"
    if args.private:
        env[-1] += " --private"
    return "bash -lc " + shlex.quote("; ".join(env))


def stop_instance(args: argparse.Namespace) -> None:
    log("Stopping GCP worker")
    run_cmd(
        [
            "/home/foivos/google-cloud-sdk/bin/gcloud",
            "compute",
            "instances",
            "stop",
            args.instance_name,
            "--project",
            args.project,
            "--zone",
            args.zone,
            "--quiet",
        ]
    )
    while True:
        proc = run_cmd(
            [
                "/home/foivos/google-cloud-sdk/bin/gcloud",
                "compute",
                "instances",
                "describe",
                args.instance_name,
                "--project",
                args.project,
                "--zone",
                args.zone,
                "--format=value(status)",
            ]
        )
        status = proc.stdout.strip()
        log(f"Instance status after stop request: {status}")
        if status == "TERMINATED":
            return
        time.sleep(30)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wait for the two continuous tokenizer runs, publish all four tokenizer arms, then stop the GCP worker.")
    parser.add_argument("--instance-name", default="apertus-greek-tokenizer-20260408t160000z")
    parser.add_argument("--project", default="eellak-glossapi-20251008")
    parser.add_argument("--zone", default="europe-west4-b")
    parser.add_argument("--worker-repo-root", default="/home/foivos/Projects/glossapi-tokenizer-extension")
    parser.add_argument("--worker-python", default="/home/foivos/venvs/tokenizer-training/bin/python")
    parser.add_argument("--repo-slug", default="apertus-tokenizer-extension")
    parser.add_argument("--repo-id", default=None)
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=900)
    parser.add_argument("--publish-stage-dir", default="/home/foivos/data/glossapi_work/tokenizer_publish_stage/apertus_tokenizer_extension")
    parser.add_argument("--fresh-glossapi-only-run-dir", default="/home/foivos/data/glossapi_work/tokenizer_training_runs_20260413/glossapi_only_50k")
    parser.add_argument("--fresh-mix-run-dir", default="/home/foivos/data/glossapi_work/tokenizer_training_runs_20260413/glossapi_plus_hplt_70_30_50k")
    parser.add_argument("--continuous-glossapi-only-run-dir", default="/home/foivos/data/glossapi_work/tokenizer_training_runs_20260415/continuous_glossapi_only_156672")
    parser.add_argument("--continuous-mix-run-dir", default="/home/foivos/data/glossapi_work/tokenizer_training_runs_20260415/continuous_glossapi_plus_hplt_70_30_156672")
    parser.add_argument("--continuous-glossapi-only-unit", default="apertus-continuous-glossapi156672-20260415.service")
    parser.add_argument("--continuous-mix-unit", default="apertus-continuous-mix156672-20260415.service")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log("Watcher starting")
    while True:
        status = remote_status(args)
        log(status_line(status))
        if any_failed(status):
            raise RuntimeError(f"Continuous tokenizer run failed or exited early: {json.dumps(status, indent=2)}")
        if all_complete(status):
            break
        time.sleep(args.poll_seconds)

    log("Both continuous runs completed; starting publish")
    publish_cmd = build_publish_command(args)
    proc = remote_shell(args.instance_name, args.project, args.zone, publish_cmd)
    if proc.stdout.strip():
        log(proc.stdout.strip())
    if proc.stderr.strip():
        log(proc.stderr.strip())
    log("Publish finished successfully; stopping worker")
    stop_instance(args)
    log("Watcher completed successfully")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log(f"Watcher failed: {exc}")
        sys.exit(1)
