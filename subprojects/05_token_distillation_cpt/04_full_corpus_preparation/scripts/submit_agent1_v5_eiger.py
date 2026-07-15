#!/usr/bin/env python3
"""Submit and babysit the receipt-bound Agent-1 v5 CPU DAG."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


TERMINAL_SUCCESS = {"COMPLETED"}
TERMINAL_FAILURE = {
    "BOOT_FAIL",
    "CANCELLED",
    "DEADLINE",
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "REVOKED",
    "SPECIAL_EXIT",
    "TIMEOUT",
}


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(f".{path.name}.partial-{os.getpid()}")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.chmod(temporary, 0o600)
    temporary.replace(path)


def command_output(*command: str) -> str:
    try:
        return subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ).stdout.strip()
    except subprocess.CalledProcessError as error:
        stderr = (error.stderr or "").strip()
        raise RuntimeError(
            f"command failed ({error.returncode}): {' '.join(command)}\n{stderr}"
        ) from error


def root_job_state(job_id: str) -> tuple[str, str]:
    output = command_output(
        "sacct",
        "-j",
        job_id,
        "--noheader",
        "--parsable2",
        "--format=JobIDRaw,State,ExitCode",
    )
    for line in output.splitlines():
        fields = line.split("|")
        if fields and fields[0] == job_id:
            return fields[1].split()[0].split("+")[0], fields[2]
    queued = subprocess.run(
        ["squeue", "-h", "-j", job_id, "-o", "%T"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    ).stdout.strip()
    return (queued.splitlines()[0] if queued else "UNKNOWN"), ""


def wait_for_jobs(job_ids: Sequence[str], poll_seconds: int) -> dict[str, tuple[str, str]]:
    pending = set(job_ids)
    final: dict[str, tuple[str, str]] = {}
    last_summary = None
    while pending:
        current = {}
        for job_id in sorted(pending, key=int):
            state, exit_code = root_job_state(job_id)
            current[job_id] = state
            if state in TERMINAL_SUCCESS | TERMINAL_FAILURE:
                final[job_id] = (state, exit_code)
        summary = tuple(sorted(Counter(current.values()).items()))
        if summary != last_summary:
            print(json.dumps({"at": utc_now(), "states": dict(summary)}, sort_keys=True), flush=True)
            last_summary = summary
        failures = {job: final[job] for job in final if final[job][0] in TERMINAL_FAILURE}
        if failures:
            raise RuntimeError(f"Slurm DAG failed: {failures}")
        pending -= set(final)
        if pending:
            time.sleep(poll_seconds)
    return final


from collections import Counter  # noqa: E402  (kept by state reporter)


def bundle_batches(task_count: int, tasks_per_node: int, maximum_elements: int) -> list[tuple[int, int]]:
    if task_count <= 0 or tasks_per_node <= 0 or maximum_elements <= 0:
        raise ValueError("bundle layout values must be positive")
    bundles = math.ceil(task_count / tasks_per_node)
    return [
        (start, min(start + maximum_elements - 1, bundles - 1))
        for start in range(0, bundles, maximum_elements)
    ]


class Submitter:
    def __init__(self, args: argparse.Namespace, config: Mapping[str, Any]):
        self.args = args
        self.config = config
        cluster = str(config["execution"]["cluster"])
        if cluster == "clariden":
            self.stage_script = args.pipeline_root / "slurm" / "agent1_v5_eiger" / "clariden_debug_stage.sh"
            self.bundle_script = args.pipeline_root / "slurm" / "agent1_v5_eiger" / "clariden_debug_bundle.sh"
        else:
            self.stage_script = args.pipeline_root / "slurm" / "agent1_v5_eiger" / "stage.sh"
            self.bundle_script = args.pipeline_root / "slurm" / "agent1_v5_eiger" / "bundle.sh"
        self.coord_root = args.run_root.parent / f".{args.run_id}.coord"
        self.log_root = self.coord_root / "slurm"
        self.runtime_root = self.coord_root / "runtime"
        self.venv_root = self.runtime_root / "venv"
        self.glossapi_root = self.runtime_root / "glossapi"
        self.datatrove_root = self.runtime_root / "datatrove"
        self.state_path = self.coord_root / "submission_state.json"
        self.bindings = {
            "pipeline_root": str(args.pipeline_root),
            "config_path": str(args.config),
            "config_sha256": sha256_file(args.config),
            "acquisition_receipt_path": str(args.acquisition_receipt),
            "acquisition_receipt_sha256": sha256_file(args.acquisition_receipt),
            "account": str(args.account or ""),
        }
        self.log_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        if args.resume:
            state = read_object(self.state_path)
            if state.get("run_id") != args.run_id:
                raise ValueError("resume state run_id does not match the requested run")
            if Path(str(state.get("run_root", ""))).resolve() != args.run_root:
                raise ValueError("resume state run_root does not match the requested run")
            if state.get("bindings") != self.bindings:
                raise ValueError("resume state input bindings do not match the requested run")
            jobs = state.get("jobs")
            if not isinstance(jobs, Mapping):
                raise ValueError("resume state has no jobs mapping")
            self.jobs = {
                str(name): dict(row)
                for name, row in jobs.items()
                if isinstance(row, Mapping)
            }
            if len(self.jobs) != len(jobs):
                raise ValueError("resume state contains an invalid job record")
        else:
            self.jobs: dict[str, dict[str, Any]] = {}

    def environment(self, extra: Mapping[str, object] | None = None) -> str:
        values: dict[str, object] = {
            "PIPELINE_ROOT": self.args.pipeline_root,
            "RUN_ROOT": self.args.run_root,
            "RUN_ID": self.args.run_id,
            "CONFIG": self.args.config,
            "ACQUISITION_RECEIPT": self.args.acquisition_receipt,
            "RUNTIME_ROOT": self.runtime_root,
            "VENV_ROOT": self.venv_root,
            "GLOSSAPI_ROOT": self.glossapi_root,
            "GLOSSAPI_BUNDLE": self.config["pins"]["glossapi_bundle"],
            "DATATROVE_ROOT": self.datatrove_root,
            "HF_TOKEN_FILE": self.args.hf_token_file,
            "SCRATCH_ROOT": self.args.run_root / "task-scratch",
        }
        if extra:
            values.update(extra)
        for key, value in values.items():
            if "," in str(value) or "\n" in str(value):
                raise ValueError(f"unsafe Slurm export value for {key}")
        return "ALL," + ",".join(f"{key}={value}" for key, value in values.items())

    def persist(self) -> None:
        write_atomic(
            self.state_path,
            {
                "schema_version": "agent1_v5_eiger_submission_state_v1",
                "updated_at": utc_now(),
                "run_id": self.args.run_id,
                "run_root": str(self.args.run_root),
                "bindings": self.bindings,
                "jobs": self.jobs,
            },
        )

    def submit(
        self,
        name: str,
        stage: str,
        *,
        partition: str,
        walltime: str,
        dependency: Sequence[str] = (),
        cpus: int = 8,
        array: str | None = None,
        command: Path | None = None,
        extra: Mapping[str, object] | None = None,
    ) -> str:
        if name in self.jobs:
            prior = self.jobs[name]
            expected = {
                "stage": stage,
                "partition": partition,
                "walltime": walltime,
                "dependency": list(dependency),
                "array": array,
            }
            observed = {key: prior.get(key) for key in expected}
            if observed != expected:
                raise ValueError(
                    f"resume job definition mismatch for {name}: "
                    f"expected {expected}, observed {observed}"
                )
            return str(prior["job_id"])
        sbatch = [
            "sbatch",
            "--parsable",
            "--uenv-passthrough=ignore",
            f"--job-name=a1v5-{name}",
            f"--partition={partition}",
            f"--time={walltime}",
            "--nodes=1",
            "--ntasks=1",
            f"--cpus-per-task={cpus}",
            f"--output={self.log_root}/%x-%A_%a.out",
            f"--error={self.log_root}/%x-%A_%a.err",
            f"--export={self.environment({'STAGE': stage, **(extra or {})})}",
        ]
        account = str(self.args.account or "").strip()
        if account:
            sbatch.append(f"--account={account}")
        if dependency:
            sbatch.append("--dependency=afterok:" + ":".join(dependency))
        if array:
            sbatch.append(f"--array={array}")
        sbatch.append(str(command or self.stage_script))
        job_id = command_output(*sbatch).split(";", 1)[0]
        self.jobs[name] = {
            "job_id": job_id,
            "stage": stage,
            "partition": partition,
            "walltime": walltime,
            "dependency": list(dependency),
            "array": array,
        }
        self.persist()
        print(json.dumps({"submitted": name, "job_id": job_id}, sort_keys=True), flush=True)
        return job_id

    def bundled(
        self,
        name: str,
        stage: str,
        task_count: int,
        tasks_per_node: int,
        dependency: Sequence[str],
        *,
        partition: str | None = None,
        walltime: str | None = None,
        max_parallel_nodes: int | None = None,
        extra: Mapping[str, object] | None = None,
        poll_seconds: int,
    ) -> list[str]:
        configured_maximum = int(self.config["execution"]["max_array_parallelism"])
        maximum = min(max_parallel_nodes or configured_maximum, configured_maximum)
        batches = bundle_batches(task_count, tasks_per_node, maximum)
        job_ids = []
        for batch_index, (start, end) in enumerate(batches):
            job_id = self.submit(
                f"{name}-part{batch_index:03d}",
                stage,
                partition=partition or str(self.config["execution"]["production_partition"]),
                walltime=walltime or str(self.config["execution"]["max_walltime"]),
                dependency=dependency,
                cpus=128,
                array=f"{start}-{end}%{maximum}",
                command=self.bundle_script,
                extra={
                    "TASK_COUNT": task_count,
                    "TASKS_PER_NODE": tasks_per_node,
                    "TASK_CONCURRENCY": tasks_per_node,
                    **(extra or {}),
                },
            )
            wait_for_jobs([job_id], poll_seconds)
            job_ids.append(job_id)
        return job_ids


def submit_dag(args: argparse.Namespace) -> dict[str, Any]:
    config = read_object(args.config)
    submitter = Submitter(args, config)
    execution = config["execution"]
    compute_partition = str(execution["production_partition"])
    transfer_partition = str(execution["transfer_partition"])
    maximum_walltime = str(execution["max_walltime"])
    if submitter.state_path.exists() and not args.resume:
        raise FileExistsError(
            f"submission state already exists: {submitter.state_path}; use --resume"
        )
    setup = submitter.submit("setup", "setup", partition=compute_partition, walltime=maximum_walltime, cpus=32)
    wait_for_jobs([setup], args.poll_seconds)
    bootstrap = submitter.submit(
        "bootstrap",
        "bootstrap",
        partition=compute_partition,
        walltime="00:30:00",
        dependency=[setup],
        cpus=8,
    )
    wait_for_jobs([bootstrap], args.poll_seconds)
    transform_tasks = int(read_object(args.run_root / "transform_tasks.json")["task_count"])
    base_tasks = int(read_object(args.run_root / "base_tasks.json")["task_count"])
    combined_tasks = transform_tasks + base_tasks
    verifier_tasks = int(config["execution"]["jaccard_verifier_tasks"])
    bucket_tasks = int(config["dedup"]["num_buckets"])

    transform_canary = submitter.submit(
        "transform-canary", "transform", partition=compute_partition, walltime="00:30:00", dependency=[bootstrap], extra={"TASK_INDEX": 0}
    )
    wait_for_jobs([transform_canary], args.poll_seconds)
    transform = submitter.bundled("transform", "transform", transform_tasks, 32, [transform_canary], poll_seconds=args.poll_seconds)
    merge_transform = submitter.submit(
        "merge-transform", "merge-transform", partition=compute_partition, walltime="00:30:00", dependency=transform
    )
    wait_for_jobs([merge_transform], args.poll_seconds)
    glossapi_canary = submitter.submit(
        "glossapi-canary", "glossapi", partition=compute_partition, walltime="00:30:00", dependency=[merge_transform], cpus=16, extra={"TASK_INDEX": 0, "GLOSSAPI_THREADS": 8}
    )
    wait_for_jobs([glossapi_canary], args.poll_seconds)
    glossapi = submitter.bundled("glossapi", "glossapi", transform_tasks, 8, [glossapi_canary], poll_seconds=args.poll_seconds)
    merge_glossapi = submitter.submit(
        "merge-glossapi", "merge-glossapi", partition=compute_partition, walltime="00:30:00", dependency=glossapi
    )
    wait_for_jobs([merge_glossapi], args.poll_seconds)
    envelope_plan = submitter.submit(
        "plan-envelope", "plan-envelope", partition=compute_partition, walltime=maximum_walltime, dependency=[merge_glossapi], cpus=16
    )
    wait_for_jobs([envelope_plan], args.poll_seconds)
    envelope = submitter.bundled("envelope", "envelope", transform_tasks, 16, [envelope_plan], poll_seconds=args.poll_seconds)
    merge_envelope = submitter.submit(
        "merge-envelope", "merge-envelope", partition=compute_partition, walltime="00:30:00", dependency=envelope
    )
    wait_for_jobs([merge_envelope], args.poll_seconds)

    base_canary = submitter.submit(
        "base-canary", "base", partition=compute_partition, walltime="00:30:00", dependency=[bootstrap], extra={"TASK_INDEX": 0}
    )
    wait_for_jobs([base_canary], args.poll_seconds)
    base = submitter.bundled("base", "base", base_tasks, 16, [base_canary], poll_seconds=args.poll_seconds)
    merge_base = submitter.submit(
        "merge-base", "merge-base", partition=compute_partition, walltime="00:30:00", dependency=base
    )
    wait_for_jobs([merge_base], args.poll_seconds)
    combine = submitter.submit(
        "combine", "combine", partition=compute_partition, walltime="00:30:00", dependency=[merge_envelope, merge_base]
    )
    wait_for_jobs([combine], args.poll_seconds)
    publish_pre = submitter.submit(
        "publish-pre", "publish-pre", partition=transfer_partition, walltime=maximum_walltime, dependency=[combine], cpus=8
    )
    wait_for_jobs([publish_pre], args.poll_seconds)

    exact_canary = submitter.submit(
        "exact-canary", "exact-index", partition=compute_partition, walltime="00:30:00", dependency=[publish_pre], extra={"TASK_INDEX": 0}
    )
    wait_for_jobs([exact_canary], args.poll_seconds)
    exact = submitter.bundled("exact", "exact-index", combined_tasks, 16, [exact_canary], poll_seconds=args.poll_seconds)
    merge_exact = submitter.submit(
        "merge-exact", "merge-exact", partition=compute_partition, walltime="00:30:00", dependency=exact
    )
    wait_for_jobs([merge_exact], args.poll_seconds)
    signature_canary = submitter.submit(
        "signature-canary", "signature", partition=compute_partition, walltime="00:30:00", dependency=[publish_pre], cpus=16, extra={"TASK_INDEX": 0}
    )
    wait_for_jobs([signature_canary], args.poll_seconds)
    signature = submitter.bundled("signature", "signature", combined_tasks, 8, [signature_canary], poll_seconds=args.poll_seconds)
    merge_signatures = submitter.submit(
        "merge-signatures", "merge-signatures", partition=compute_partition, walltime="00:30:00", dependency=signature
    )
    wait_for_jobs([merge_signatures], args.poll_seconds)
    bucket_canary = submitter.submit(
        "bucket-canary", "bucket", partition=compute_partition, walltime="00:30:00", dependency=[merge_signatures], cpus=32, extra={"TASK_INDEX": 0}
    )
    wait_for_jobs([bucket_canary], args.poll_seconds)
    buckets = submitter.bundled("buckets", "bucket", bucket_tasks, 1, [bucket_canary], max_parallel_nodes=32, poll_seconds=args.poll_seconds)
    pairs = submitter.submit(
        "merge-pairs", "merge-pairs", partition=compute_partition, walltime=maximum_walltime, dependency=buckets, cpus=32
    )
    wait_for_jobs([pairs], args.poll_seconds)
    shingles = submitter.bundled("shingles", "shingles", combined_tasks, 8, [pairs], poll_seconds=args.poll_seconds)
    merge_shingles = submitter.submit(
        "merge-shingles", "merge-shingles", partition=compute_partition, walltime="00:30:00", dependency=shingles
    )
    wait_for_jobs([merge_shingles], args.poll_seconds)
    verify_canary = submitter.submit(
        "verify-canary", "verify", partition=compute_partition, walltime="00:30:00", dependency=[merge_shingles], cpus=32, extra={"TASK_INDEX": 0, "VERIFY_TASKS": verifier_tasks}
    )
    wait_for_jobs([verify_canary], args.poll_seconds)
    verify = submitter.bundled(
        "verify",
        "verify",
        verifier_tasks,
        32,
        [verify_canary],
        max_parallel_nodes=128,
        extra={"VERIFY_TASKS": verifier_tasks},
        poll_seconds=args.poll_seconds,
    )
    merge_verified = submitter.submit(
        "merge-verified", "merge-verified", partition=compute_partition, walltime="00:30:00", dependency=verify, extra={"VERIFY_TASKS": verifier_tasks}
    )
    wait_for_jobs([merge_verified], args.poll_seconds)
    cluster = submitter.submit(
        "cluster", "cluster", partition=compute_partition, walltime=maximum_walltime, dependency=[merge_exact, merge_verified], cpus=64
    )
    wait_for_jobs([cluster], args.poll_seconds)
    filter_job = submitter.bundled("filter", "filter", combined_tasks, 16, [cluster], poll_seconds=args.poll_seconds)
    merge_filtered = submitter.submit(
        "merge-filtered", "merge-filtered", partition=compute_partition, walltime="00:30:00", dependency=filter_job
    )
    wait_for_jobs([merge_filtered], args.poll_seconds)
    publish_dedup = submitter.submit(
        "publish-dedup", "publish-dedup", partition=transfer_partition, walltime=maximum_walltime, dependency=[merge_filtered], cpus=8
    )
    wait_for_jobs([publish_dedup], args.poll_seconds)
    submitter.persist()
    graph_path = args.run_root / "slurm_job_graph.json"
    write_atomic(
        graph_path,
        {
            "schema_version": "agent1_v5_eiger_job_graph_v1",
            "status": "completed",
            "created_at": utc_now(),
            "completed_at": utc_now(),
            "run_id": args.run_id,
            "counts": {
                "transform_tasks": transform_tasks,
                "base_tasks": base_tasks,
                "combined_tasks": combined_tasks,
                "bucket_tasks": bucket_tasks,
                "verifier_tasks": verifier_tasks,
            },
            "jobs": submitter.jobs,
            "terminal_job_id": publish_dedup,
        },
    )
    return read_object(graph_path)


def monitor_state(state_path: Path, poll_seconds: int) -> dict[str, tuple[str, str]]:
    state = read_object(state_path)
    jobs = state.get("jobs")
    if not isinstance(jobs, Mapping) or not jobs:
        raise ValueError("submission state has no jobs")
    return wait_for_jobs([str(row["job_id"]) for row in jobs.values()], poll_seconds)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline-root", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--acquisition-receipt", type=Path)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--hf-token-file", type=Path)
    parser.add_argument("--account", default=None)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume the receipt-bound DAG from its persisted submission state",
    )
    parser.add_argument("--monitor", action="store_true")
    parser.add_argument("--monitor-state", type=Path)
    args = parser.parse_args(argv)
    if args.monitor_state:
        return args
    required = ("pipeline_root", "config", "acquisition_receipt", "run_root", "run_id", "hf_token_file")
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        parser.error(f"submission requires: {', '.join(missing)}")
    args.pipeline_root = args.pipeline_root.resolve()
    args.config = args.config.resolve()
    args.acquisition_receipt = args.acquisition_receipt.resolve()
    args.run_root = args.run_root.resolve()
    args.hf_token_file = args.hf_token_file.resolve()
    state_path = args.run_root.parent / f".{args.run_id}.coord" / "submission_state.json"
    if args.resume and not state_path.is_file():
        parser.error(f"resume state does not exist: {state_path}")
    if args.run_root.exists() and not args.resume:
        parser.error(f"run root must not exist: {args.run_root}")
    if not args.hf_token_file.is_file() or (args.hf_token_file.stat().st_mode & 0o077):
        parser.error("Hugging Face token file must exist and be mode 600")
    if args.account is None:
        args.account = read_object(args.config)["execution"].get("account")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.monitor_state:
        final = monitor_state(args.monitor_state.resolve(), args.poll_seconds)
        print(json.dumps({"ok": True, "final": final}, sort_keys=True))
        return 0
    graph = submit_dag(args)
    if args.monitor:
        state_path = args.run_root.parent / f".{args.run_id}.coord" / "submission_state.json"
        final = monitor_state(state_path, args.poll_seconds)
        graph["status"] = "completed"
        graph["completed_at"] = utc_now()
        graph["final_states"] = final
        write_atomic(args.run_root / "slurm_job_graph.completed.json", graph)
    print(json.dumps({"ok": True, "jobs": len(graph["jobs"]), "monitor": args.monitor}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
