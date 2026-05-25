#!/usr/bin/env python3
"""Incrementally submit 3.5B continuation eval sidecars under Slurm limits.

The all-at-once submitter is ideal when the account can hold the full eval DAG
in Slurm. Clariden currently rejects that many submitted jobs for this user, so
this helper records already-submitted sidecars and trickles in the rest as
slots open. Training jobs remain independent; eval jobs depend on the
checkpoint-producing training jobs or conversion jobs only.
"""

import csv
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple


ARMS = tuple(os.environ.get("EVAL_ARMS", "vanilla retok td_layer11").split())
DIAG_ARMS = tuple(os.environ.get("DIAG_ARMS", "retok td_layer11").split())
PACKED_JOB_PREFIX = os.environ.get("PACKED_JOB_PREFIX", "eval_3p5")


class TrainRow(NamedTuple):
    run_tag: str
    segment: str
    target_iter: str
    target_tokens: str
    arm: str
    train_arm: str
    output_dir: str
    init_ckpt: str
    dependency_job: str
    job_id: str


class Task(NamedTuple):
    task_id: str
    kind: str
    iteration: str
    arm: str
    job_name: str
    deps: Tuple[str, ...]
    command: Tuple[str, ...]


def run(cmd, check=True):
    return subprocess.run(cmd, universal_newlines=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=check)


def env(name, default=None):
    value = os.environ.get(name, default)
    if value is None or value == "":
        raise SystemExit(f"ERROR: {name} is required")
    return value


def load_training_chain(path):
    rows = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            tr = TrainRow(**{key: row[key] for key in TrainRow._fields})
            rows[(tr.target_iter, tr.arm)] = tr
    return rows


def load_training_rows(primary_path, overlay_paths):
    rows = load_training_chain(primary_path)
    for raw_path in overlay_paths.split(":"):
        raw_path = raw_path.strip()
        if not raw_path:
            continue
        rows.update(load_training_chain(Path(raw_path)))
    return rows


def load_state(path):
    if not path.exists():
        return {}
    state = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row.get("task_id") and row.get("job_id"):
                state[row["task_id"]] = row["job_id"]
    return state


def write_state(path, tasks, state):
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["task_id", "kind", "iter", "arm", "job_name", "job_id"])
        for task in tasks:
            if task.task_id in state:
                writer.writerow([
                    task.task_id,
                    task.kind,
                    task.iteration,
                    task.arm,
                    task.job_name,
                    state[task.task_id],
                ])
    tmp.replace(path)


def squeue_jobs():
    proc = run(["squeue", "-u", env("USER", "fffoivos"), "-h", "-o", "%i|%j"], check=False)
    jobs = {}
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
        return jobs
    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        job_id, name = line.split("|", 1)
        jobs.setdefault(name, []).append(job_id)
    return jobs


def active_job_count():
    proc = run(["squeue", "-u", env("USER", "fffoivos"), "-h", "-o", "%i"], check=False)
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
        return 999999
    return len([line for line in proc.stdout.splitlines() if line.strip()])


def dep_arg(job_ids):
    return "--dependency=afterok:" + ":".join(job_ids)


def active_job_ids(job_ids):
    if not job_ids:
        return set()
    # Query the user's current queue and intersect locally. `squeue -j <old_id>`
    # returns an error once a completed job leaves the live queue, but that is
    # exactly the case where we should fall through to `sacct` below.
    proc = run(["squeue", "-u", env("USER", "fffoivos"), "-h", "-o", "%i"], check=False)
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
        return set()
    live_ids = set(line.strip().split()[0] for line in proc.stdout.splitlines() if line.strip())
    return set(job_ids) & live_ids


def completed_ok(job_id):
    proc = run(["sacct", "-X", "-n", "-P", "-j", job_id, "--format=State,ExitCode"], check=False)
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
        return False
    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        state, exit_code = (line.split("|", 1) + [""])[:2]
        if state == "COMPLETED" and exit_code == "0:0":
            return True
    return False


def unresolved_dependency_ids(job_ids):
    active = active_job_ids(job_ids)
    unresolved = []
    for job_id in job_ids:
        if job_id in active:
            unresolved.append(job_id)
        elif not completed_ok(job_id):
            unresolved.append(job_id)
    return tuple(unresolved)


def build_tasks(
    *,
    run_tag,
    training_rows,
    iter_list,
    task_group,
    out_root,
    eval_jsonl,
    nice,
    script_dir,
    state_dir,
    overwrite,
):
    tasks = []
    nice_args = (f"--nice={nice}",) if nice else ()

    for iteration in iter_list:
        iter_pad = f"{int(iteration):07d}"
        iter_state_dir = state_dir / f"iter_{iter_pad}_{task_group}"
        iter_state_dir.mkdir(parents=True, exist_ok=True)
        spec_tsv = iter_state_dir / "eval_spec.tsv"
        with spec_tsv.open("w") as f:
            for arm in ARMS:
                hf_out_dir = f"{out_root}/{run_tag}_{arm}/iter_{iter_pad}_hf"
                eval_out_dir = f"{out_root}/{run_tag}_{arm}/iter_{iter_pad}_{task_group}"
                f.write(f"{arm}\t{hf_out_dir}\t{eval_out_dir}\n")

        for arm in ARMS:
            train = training_rows[(iteration, arm)]
            hf_out_dir = f"{out_root}/{run_tag}_{arm}/iter_{iter_pad}_hf"
            convert_id = f"convert:{iteration}:{arm}"
            tasks.append(Task(
                task_id=convert_id,
                kind="convert",
                iteration=iteration,
                arm=arm,
                job_name=f"tohf_{arm}_{iteration}",
                deps=(train.job_id,),
                command=(
                    "sbatch",
                    "--parsable",
                    "__DEPENDENCY_PLACEHOLDER__",
                    *nice_args,
                    (
                        "--export=ALL,"
                        f"RUN_TAG={run_tag},ARM={arm},ITER={iteration},"
                        f"MEGATRON_CKPT_ROOT={train.output_dir}/checkpoints,"
                        f"HF_OUT_DIR={hf_out_dir},OUT_ROOT={out_root},"
                        f"SCRIPT_DIR_OVERRIDE={script_dir},OVERWRITE={overwrite}"
                    ),
                    f"--job-name=tohf_{arm}_{iteration}",
                    str(script_dir / "convert_bakeoff_checkpoint_to_hf.sbatch"),
                ),
            ))

            tasks.append(Task(
                task_id=f"bpc:{iteration}:{arm}",
                kind="bpc",
                iteration=iteration,
                arm=arm,
                job_name=f"bpc_{arm}_{iteration}",
                deps=(convert_id,),
                command=(
                    "sbatch",
                    "--parsable",
                    "__DEPENDENCY_PLACEHOLDER__",
                    *nice_args,
                    (
                        "--export=ALL,"
                        f"MODEL_PATH={hf_out_dir},EVAL_JSONL={eval_jsonl},"
                        f"OUTPUT_JSON={out_root}/{run_tag}_{arm}/iter_{iter_pad}_tokenizer_fair_metrics.json,"
                        f"SCRIPT_DIR_OVERRIDE={script_dir},OVERWRITE={overwrite}"
                    ),
                    f"--job-name=bpc_{arm}_{iteration}",
                    str(script_dir / "run_tokenizer_fair_metrics.sbatch"),
                ),
            ))

            if arm in DIAG_ARMS:
                tasks.append(Task(
                    task_id=f"diag:{iteration}:{arm}",
                    kind="diag",
                    iteration=iteration,
                    arm=arm,
                    job_name=f"diag_{arm}_{iteration}",
                    deps=(convert_id,),
                    command=(
                        "sbatch",
                        "--parsable",
                        "__DEPENDENCY_PLACEHOLDER__",
                        *nice_args,
                        (
                            "--export=ALL,"
                            f"MODEL_PATH={hf_out_dir},EVAL_JSONL={eval_jsonl},"
                            f"OUTPUT_JSON={out_root}/{run_tag}_{arm}/iter_{iter_pad}_new_token_diagnostics.json,"
                            f"SCRIPT_DIR_OVERRIDE={script_dir},OVERWRITE={overwrite}"
                        ),
                        f"--job-name=diag_{arm}_{iteration}",
                        str(script_dir / "run_new_token_diagnostics.sbatch"),
                    ),
                ))

        convert_ids = tuple(f"convert:{iteration}:{arm}" for arm in ARMS)
        tasks.append(Task(
            task_id=f"packed:{iteration}:{task_group}",
            kind="packed",
            iteration=iteration,
            arm="all",
            job_name=f"{PACKED_JOB_PREFIX}_{iteration}_{task_group}",
            deps=convert_ids,
            command=(
                "sbatch",
                "--parsable",
                "__DEPENDENCY_PLACEHOLDER__",
                *nice_args,
                f"--export=ALL,EVAL_SPEC_TSV={spec_tsv},TASK_GROUP={task_group}",
                f"--job-name={PACKED_JOB_PREFIX}_{iteration}_{task_group}",
                str(script_dir / "run_eval_packed_arms.sbatch"),
            ),
        ))

    return tasks


def concrete_command(task, state):
    dep_job_ids = []
    for dep in task.deps:
        if dep.isdigit():
            dep_job_ids.append(dep)
        elif dep in state:
            dep_job_ids.append(state[dep])
        else:
            return None
    unresolved = unresolved_dependency_ids(tuple(dep_job_ids))
    dependency = dep_arg(unresolved) if unresolved else None
    parts = []
    for part in task.command:
        if part == "__DEPENDENCY_PLACEHOLDER__":
            if dependency:
                parts.append(dependency)
        else:
            parts.append(part)
    return tuple(parts)


def submit_once(tasks, state_path, command_log, max_jobs):
    state = load_state(state_path)
    current_by_name = squeue_jobs()

    # Import jobs that were submitted by an earlier interrupted all-at-once run.
    for task in tasks:
        if task.task_id not in state and task.job_name in current_by_name:
            state[task.task_id] = sorted(current_by_name[task.job_name])[0]
    write_state(state_path, tasks, state)

    made_progress = False
    for task in tasks:
        if task.task_id in state:
            continue
        if active_job_count() >= max_jobs:
            break
        cmd = concrete_command(task, state)
        if cmd is None:
            continue
        with command_log.open("a") as f:
            f.write(" ".join(subprocess.list2cmdline([part]) for part in cmd) + "\n")
        proc = run(cmd, check=False)
        if proc.returncode != 0:
            print(f"submit failed for {task.task_id}: {proc.stderr.strip()}", file=sys.stderr)
            break
        job_id = proc.stdout.strip().split(";", 1)[0]
        state[task.task_id] = job_id
        write_state(state_path, tasks, state)
        print(f"submitted {task.task_id} -> {job_id}")
        made_progress = True

    missing = [task.task_id for task in tasks if task.task_id not in state]
    print(f"state: submitted={len(tasks) - len(missing)} missing={len(missing)} active_jobs={active_job_count()}")
    if missing:
        print("next_missing:", ", ".join(missing[:8]))
    return made_progress or not missing


def main() -> int:
    script_dir = Path(os.environ.get("SCRIPT_DIR_OVERRIDE", Path(__file__).resolve().parent)).resolve()
    run_tag = env("RUN_TAG")
    training_chain_tsv = Path(env("TRAINING_CHAIN_TSV"))
    training_chain_overlay_tsv = os.environ.get("TRAINING_CHAIN_OVERLAY_TSV", "")
    out_root = env("OUT_ROOT", "/capstor/scratch/cscs/fffoivos/runs/eval")
    state_dir = Path(env("STATE_DIR", f"{out_root}/{run_tag}_sidecar_eval_incremental"))
    state_dir.mkdir(parents=True, exist_ok=True)

    tasks = build_tasks(
        run_tag=run_tag,
        training_rows=load_training_rows(training_chain_tsv, training_chain_overlay_tsv),
        iter_list=env("ITER_LIST", "585 715 834").split(),
        task_group=env("TASK_GROUP", "full"),
        out_root=out_root,
        eval_jsonl=env("EVAL_JSONL", "/iopsstor/scratch/cscs/fffoivos/cpt_corpus/heldout/cpt_greek_heldout_500_20260522.jsonl"),
        nice=env("EVAL_NICE", "1000"),
        script_dir=script_dir,
        state_dir=state_dir,
        overwrite=env("OVERWRITE_EVAL", "0"),
    )

    state_path = state_dir / "eval_sidecar_incremental_state.tsv"
    command_log = state_dir / "eval_sidecar_incremental_commands.sh"
    max_jobs = int(env("MAX_SUBMITTED_JOBS", "14"))
    sleep_seconds = int(env("SLEEP_SECONDS", "120"))
    loop = env("LOOP", "0") == "1"

    while True:
        done_or_progress = submit_once(tasks, state_path, command_log, max_jobs)
        state = load_state(state_path)
        if len(state) == len(tasks):
            print(f"all {len(tasks)} sidecar tasks submitted")
            return 0
        if not loop:
            return 0 if done_or_progress else 1
        time.sleep(sleep_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
