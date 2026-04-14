#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage a prepared release snapshot for the cheap uploader instance and optionally launch the detached HF upload."
    )
    parser.add_argument("--handoff-json", type=Path, required=True)
    parser.add_argument("--ssh-key", default="/home/foivos/.ssh/glossapi_hf_box")
    parser.add_argument("--local-stage-root", type=Path, default=None)
    parser.add_argument("--skip-sync", action="store_true")
    parser.add_argument("--skip-launch", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summary-json", type=Path, default=None)
    return parser.parse_args()


def run_command(cmd: list[str], *, dry_run: bool) -> subprocess.CompletedProcess[str] | None:
    if dry_run:
        return None
    return subprocess.run(cmd, check=True, text=True, capture_output=True)


def local_stage_copy(source_root: Path, dest_root: Path) -> None:
    if dest_root.exists():
        shutil.rmtree(dest_root)
    shutil.copytree(source_root, dest_root)


def main() -> None:
    args = parse_args()
    handoff_json = args.handoff_json.resolve()
    manifest = json.loads(handoff_json.read_text(encoding="utf-8"))

    source_root = Path(str(manifest["working_release_root"])).resolve()
    host = str(manifest["dataset_server"]["host"])
    user = str(manifest["dataset_server"]["user"])
    remote_release_root = str(manifest["dataset_server"]["remote_release_root"])
    remote_detach_command = str(manifest["upload"]["remote_detach_command"])

    if not source_root.exists():
        raise SystemExit(f"Missing source release root: {source_root}")

    commands: dict[str, str] = {}
    staged_release_root: str | None = None
    ssh_base = [
        "ssh",
        "-i",
        args.ssh_key,
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=accept-new",
    ]
    rsync_ssh = " ".join(
        [
            "ssh",
            "-i",
            shlex.quote(args.ssh_key),
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=accept-new",
        ]
    )

    if not args.skip_sync:
        if args.local_stage_root is not None:
            local_stage_root = args.local_stage_root.resolve()
            staged_release_root = str((local_stage_root / Path(remote_release_root).name).resolve())
            commands["local_stage_copy"] = f"copytree {source_root} -> {staged_release_root}"
            if not args.dry_run:
                local_stage_root.mkdir(parents=True, exist_ok=True)
                local_stage_copy(source_root, Path(staged_release_root))
        else:
            ssh_mkdir = ssh_base + [
                f"{user}@{host}",
                f"mkdir -p {shlex.quote(remote_release_root)}",
            ]
            rsync_cmd = []
            if shutil.which("ionice"):
                rsync_cmd.extend(["ionice", "-c2", "-n7"])
            if shutil.which("nice"):
                rsync_cmd.extend(["nice", "-n", "10"])
            rsync_cmd.extend([
                "rsync",
                "-a",
                "--delete",
                "-e",
                rsync_ssh,
                f"{source_root}/",
                f"{user}@{host}:{remote_release_root}/",
            ])
            commands["ssh_mkdir"] = shlex.join(ssh_mkdir)
            commands["rsync"] = shlex.join(rsync_cmd)
            run_command(ssh_mkdir, dry_run=args.dry_run)
            run_command(rsync_cmd, dry_run=args.dry_run)

    if not args.skip_launch:
        if args.local_stage_root is not None:
            local_launch_note = Path(staged_release_root or str(args.local_stage_root.resolve())) / "planned_remote_upload_command.txt"
            commands["local_launch_note"] = str(local_launch_note)
            if not args.dry_run:
                local_launch_note.parent.mkdir(parents=True, exist_ok=True)
                local_launch_note.write_text(remote_detach_command + "\n", encoding="utf-8")
        else:
            ssh_launch = ssh_base + [f"{user}@{host}", remote_detach_command]
            commands["ssh_launch"] = shlex.join(ssh_launch)
            run_command(ssh_launch, dry_run=args.dry_run)

    summary = {
        "handoff_json": str(handoff_json),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dry_run": bool(args.dry_run),
        "skip_sync": bool(args.skip_sync),
        "skip_launch": bool(args.skip_launch),
        "local_stage_root": str(args.local_stage_root.resolve()) if args.local_stage_root else None,
        "staged_release_root": staged_release_root,
        "commands": commands,
        "sync_executed": not args.skip_sync and not args.dry_run,
        "launch_executed": not args.skip_launch and not args.dry_run and args.local_stage_root is None,
    }
    summary_path = args.summary_json.resolve() if args.summary_json else handoff_json.with_name("launch_summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
