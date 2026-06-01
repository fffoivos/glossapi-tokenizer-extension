#!/usr/bin/env python3
"""Collect structured state for the Vanilla 5B CPT report.

Run from home. The script reads Clariden artifacts through SSH and augments the
remote state with local adversarial-review artifacts. It does not submit,
cancel, or modify remote jobs.
"""

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Dict


DEFAULT_RUN_TAG = "04_vanilla_goldfish_5b_20260528T112539Z"
DEFAULT_RUN_ROOT = "/capstor/scratch/cscs/fffoivos/runs/04_vanilla_cpt"
DEFAULT_CANONICAL_TASKS = (
    Path(__file__).resolve().parents[1] / "goal" / "canonical_eval_tasks.json"
)
CHECKPOINTS = [
    {"iteration": 119, "tokens": 499_122_176, "label": "Vanilla-0.5B"},
    {"iteration": 238, "tokens": 998_244_352, "label": "Vanilla-1B"},
    {"iteration": 477, "tokens": 2_000_683_008, "label": "Vanilla-2B"},
    {"iteration": 834, "tokens": 3_498_049_536, "label": "Vanilla-3.5B"},
    {"iteration": 1192, "tokens": 4_999_610_368, "label": "Vanilla-5B"},
]


REMOTE_PROGRAM = r"""
import csv
import datetime as dt
import json
import os
from pathlib import Path
import re
import subprocess

run_tag = os.environ["RUN_TAG"]
run_root = Path(os.environ["RUN_ROOT"])
train_run_dir = run_root / run_tag
eval_root = run_root / f"eval_{run_tag}"
watch_dir = run_root / f"{run_tag}_sidecar_watch"
submit_state_dir = run_root / f"{run_tag}_submit_state"
active_log = run_root / "04van5b_i300-2417446.out"
watch_out = run_root / "04van5b_watch-2417454.out"
watch_err = run_root / "04van5b_watch-2417454.err"

# Canonical task lockdown — passed in verbatim from
# goal/canonical_eval_tasks.json so render-time filtering is grounded
# in a single locked source rather than prose comments.
CANONICAL_TASKS = json.loads(os.environ["CANONICAL_TASKS_JSON"])
_CANONICAL_HEADLINE = set(
    CANONICAL_TASKS.get("greek_native_mcq", {}).get("headline_tasks", [])
)
_CANONICAL_DIAGNOSTIC = set(
    CANONICAL_TASKS.get("greek_native_mcq", {}).get("diagnostic_separate", [])
)
_CANONICAL_EXCLUDED_MT = set(
    CANONICAL_TASKS.get("greek_native_mcq", {}).get("excluded_mt_derived", [])
)
_CANONICAL_RETENTION_LANG_TASKS = {}
for _lang, _entries in (
    CANONICAL_TASKS.get("retention", {}).get("tasks_per_language", {}).items()
):
    for _entry in _entries:
        if _entry.get("absent_on_disk"):
            continue
        _CANONICAL_RETENTION_LANG_TASKS.setdefault(_lang, []).append(_entry["task"])
_CANONICAL_RETENTION_ALL = sorted(
    {task for tasks in _CANONICAL_RETENTION_LANG_TASKS.values() for task in tasks}
)

checkpoints = [
    {"iteration": 119, "tokens": 499122176, "label": "Vanilla-0.5B"},
    {"iteration": 238, "tokens": 998244352, "label": "Vanilla-1B"},
    {"iteration": 477, "tokens": 2000683008, "label": "Vanilla-2B"},
    {"iteration": 834, "tokens": 3498049536, "label": "Vanilla-3.5B"},
    {"iteration": 1192, "tokens": 4999610368, "label": "Vanilla-5B"},
]


def read_text(path, limit=None):
    try:
        text = Path(path).read_text(errors="replace")
    except FileNotFoundError:
        return None
    if limit is not None and len(text) > limit:
        return text[-limit:]
    return text


def tail_lines(path, n=20):
    text = read_text(path)
    if text is None:
        return []
    return text.splitlines()[-n:]


def load_json(path):
    try:
        with Path(path).open() as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError as exc:
        return {"_error": f"json decode failed: {exc}"}


def read_tsv_rows(path):
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


def parse_latest_training_line(path):
    latest = ""
    nonzero = []
    severe = []
    severe_re = re.compile(
        r"traceback|runtimeerror|assertionerror|cuda out of memory|out of memory|\boom\b|\bfailed\b|\bfatal\b|\berror\b",
        re.I,
    )
    try:
        with Path(path).open(errors="replace") as f:
            for lineno, line in enumerate(f, 1):
                if re.search(r"iteration\s+\d+/", line):
                    latest = line.rstrip("\n")
                    skipped = re.search(r"number of skipped iterations:\s*(\d+)", line)
                    nan = re.search(r"number of nan iterations:\s*(\d+)", line)
                    if (skipped and int(skipped.group(1))) or (nan and int(nan.group(1))):
                        nonzero.append({"line": lineno, "text": line.rstrip("\n")})
                elif severe_re.search(line) and "max_tokens_to_oom" not in line:
                    severe.append({"line": lineno, "text": line.rstrip("\n")})
    except FileNotFoundError:
        return {"exists": False, "latest_line": ""}

    parsed = {"exists": True, "latest_line": latest}
    m = re.search(
        r"iteration\s+(\d+)/\s*(\d+).*consumed tokens:\s*([0-9.]+)B"
        r".*elapsed time per iteration \(ms\):\s*([0-9.]+)"
        r".*tokens/sec/gpu:\s*([0-9.]+)"
        r".*learning rate:\s*([0-9.E+-]+)"
        r".*lm loss:\s*([0-9.E+-]+)"
        r".*grad norm:\s*([0-9.]+)"
        r".*params norm:\s*([0-9.]+)"
        r".*number of skipped iterations:\s*(\d+)"
        r".*number of nan iterations:\s*(\d+)",
        latest,
    )
    if m:
        current_iter = int(m.group(1))
        iter_ms = float(m.group(4))
        now = dt.datetime.now(dt.timezone.utc)
        parsed.update(
            {
                "current_iter": current_iter,
                "target_iter": int(m.group(2)),
                "consumed_tokens_b": float(m.group(3)),
                "iter_ms": iter_ms,
                "tokens_sec_gpu": float(m.group(5)),
                "learning_rate": m.group(6),
                "lm_loss": float(m.group(7)),
                "grad_norm": float(m.group(8)),
                "params_norm": float(m.group(9)),
                "skipped": int(m.group(10)),
                "nan": int(m.group(11)),
                "etas": {
                    str(target): (
                        now
                        + dt.timedelta(seconds=max(0, target - current_iter) * iter_ms / 1000.0)
                    )
                    .replace(microsecond=0)
                    .isoformat()
                    .replace("+00:00", "Z")
                    for target in (119, 238, 300, 477, 834, 1192)
                },
            }
        )
    parsed["nonzero_skipped_or_nan_count"] = len(nonzero)
    parsed["nonzero_skipped_or_nan_tail"] = nonzero[-5:]
    parsed["severe_term_count"] = len(severe)
    parsed["severe_term_tail"] = severe[-10:]
    return parsed


def parse_training_chain(path):
    text = read_text(path)
    if not text:
        return []
    return list(csv.DictReader(text.splitlines(), delimiter="\t"))


def queue_rows():
    result = run_cmd(
        [
            "squeue",
            "-u",
            "fffoivos",
            "-h",
            "-o",
            "%i|%P|%j|%T|%M|%l|%R",
        ]
    )
    rows = []
    for line in result["stdout"].splitlines():
        parts = line.split("|", 6)
        if len(parts) != 7:
            continue
        job_id, partition, name, state, elapsed, limit, reason = parts
        if (
            name.startswith("04van5b")
            or name.startswith("04tohf")
            or name.startswith("04native")
            or name.startswith("04gnlp")
            or name.startswith("04bpb")
            or name.startswith("04ret")
            or name.startswith("04code")
            or name.startswith("04math")
            or name.startswith("04cksum")
        ):
            rows.append(
                {
                    "job_id": job_id.strip(),
                    "partition": partition.strip(),
                    "name": name.strip(),
                    "state": state.strip(),
                    "elapsed": elapsed.strip(),
                    "limit": limit.strip(),
                    "reason": reason.strip(),
                }
            )
    return rows


def count_items(root, kind):
    if not root.exists():
        return 0
    if kind == "files":
        return sum(1 for p in root.rglob("*") if p.is_file())
    if kind == "all":
        return sum(1 for p in root.rglob("*"))
    raise ValueError(kind)


def path_info(path):
    path = Path(path)
    info = {"path": str(path), "exists": path.exists()}
    if info["exists"]:
        stat = path.stat()
        info.update({"size": stat.st_size, "mtime": dt.datetime.fromtimestamp(stat.st_mtime, dt.timezone.utc).isoformat().replace("+00:00", "Z")})
    return info


def parse_env_file(path):
    text = read_text(path)
    env = {}
    if not text:
        return env
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] == '"':
            value = value[1:-1]
        env[key.strip()] = value
    return env


def archived_sidecar_attempts(iter_root):
    attempts = []
    if not iter_root.exists():
        return attempts
    for path in sorted(iter_root.glob("sidecar_jobs_attempt_*.tsv")):
        rows = read_tsv_rows(path)
        attempts.append(
            {
                "file": path_info(path),
                "rows": rows,
                "job_ids": [
                    row.get("job_id", "") for row in rows if row.get("job_id")
                ],
            }
        )
    return attempts


def watcher_env_states(root):
    rows = []
    if not root.exists():
        return rows
    for path in sorted(root.glob("watcher_*.env")):
        env = parse_env_file(path)
        rows.append(
            {
                "file": path_info(path),
                "slurm_job_id": env.get("SLURM_JOB_ID", ""),
                "start_time_utc": env.get("START_TIME_UTC", ""),
                "sleep_seconds": env.get("SLEEP_SECONDS", ""),
                "max_watch_seconds": env.get("MAX_WATCH_SECONDS", ""),
                "resubmit_if_incomplete": env.get("RESUBMIT_IF_INCOMPLETE", ""),
                "require_code_math_heldouts": env.get("REQUIRE_CODE_MATH_HELDOUTS", ""),
                "code_heldout_jsonl": env.get("CODE_HELDOUT_JSONL", ""),
                "math_heldout_jsonl": env.get("MATH_HELDOUT_JSONL", ""),
                "raw": env,
            }
        )
    return rows


def mean(values):
    values = [value for value in values if isinstance(value, (int, float))]
    if not values:
        return None
    return sum(values) / len(values)


def native_mcq_metrics(iter_root, label):
    native_root = iter_root / "native_mcq"
    headline_raw = load_json(native_root / ("%s_native_mcq_headline.json" % label))
    diagnostics_raw = load_json(native_root / ("%s_native_mcq_diagnostics.json" % label))
    headline_list = headline_raw if isinstance(headline_raw, list) else []
    diagnostics_list = diagnostics_raw if isinstance(diagnostics_raw, list) else []

    # Canonical filter: only locked headline tasks contribute to the headline
    # aggregate; only locked diagnostic tasks are reported as diagnostics; any
    # MT-derived Greek tasks listed in excluded_mt_derived are dropped from any
    # aggregate. Non-canonical data stays on disk but is invisible here.
    headline_filtered = [
        row for row in headline_list
        if row.get("benchmark") in _CANONICAL_HEADLINE
        and row.get("benchmark") not in _CANONICAL_EXCLUDED_MT
    ]
    diagnostics_filtered = [
        row for row in diagnostics_list
        if row.get("benchmark") in _CANONICAL_DIAGNOSTIC
        and row.get("benchmark") not in _CANONICAL_EXCLUDED_MT
    ]
    dropped_headline = sorted(
        {
            row.get("benchmark")
            for row in headline_list
            if row.get("benchmark") not in _CANONICAL_HEADLINE
            or row.get("benchmark") in _CANONICAL_EXCLUDED_MT
        }
    )
    dropped_diagnostics = sorted(
        {
            row.get("benchmark")
            for row in diagnostics_list
            if row.get("benchmark") not in _CANONICAL_DIAGNOSTIC
            or row.get("benchmark") in _CANONICAL_EXCLUDED_MT
        }
    )
    metrics = {
        "headline_file": path_info(native_root / ("%s_native_mcq_headline.json" % label)),
        "diagnostics_file": path_info(native_root / ("%s_native_mcq_diagnostics.json" % label)),
        "headline": headline_filtered,
        "diagnostics": diagnostics_filtered,
        "canonical_headline_tasks": sorted(_CANONICAL_HEADLINE),
        "canonical_diagnostic_tasks": sorted(_CANONICAL_DIAGNOSTIC),
        "dropped_non_canonical_headline": dropped_headline,
        "dropped_non_canonical_diagnostics": dropped_diagnostics,
    }
    metrics["headline_macro_accuracy"] = mean(
        [row.get("accuracy") for row in metrics["headline"]]
    )
    metrics["headline_plus_diagnostic_macro_accuracy"] = mean(
        [row.get("accuracy") for row in metrics["headline"] + metrics["diagnostics"]]
    )
    return metrics


def bpb_metrics(iter_root):
    results = {}
    for name, filename in [
        ("greek", "heldout_greek_bpb.json"),
        ("code", "heldout_code_bpb.json"),
        ("math", "heldout_math_bpb.json"),
    ]:
        path = iter_root / filename
        data = load_json(path)
        global_metrics = data.get("global", {}) if isinstance(data, dict) else {}
        truncation = data.get("truncation", {}) if isinstance(data, dict) else {}
        strr = data.get("strr", {}) if isinstance(data, dict) else {}
        results[name] = {
            "file": path_info(path),
            "bpb": global_metrics.get("bpb_bits_per_byte"),
            "n_docs": global_metrics.get("n_docs"),
            "fraction_truncated": truncation.get("fraction_truncated"),
            "n_docs_truncated": truncation.get("n_docs_truncated"),
            "strr_rate": strr.get("rate"),
        }
    return results


def retention_metrics(iter_root):
    retention_root = iter_root / "retention"
    result_files = sorted(retention_root.glob("results_*.json"))
    path = result_files[-1] if result_files else retention_root / "results_missing.json"
    data = load_json(path)
    task_results = data.get("results", {}) if isinstance(data, dict) else {}

    # Canonical filter: drop non-canonical retention tasks entirely from the
    # state JSON. Non-canonical data stays on disk (the upstream results_*.json
    # is untouched); the renderer just never sees it.
    canonical = sorted(_CANONICAL_RETENTION_ALL)
    selected = {}
    for task in canonical:
        row = task_results.get(task, {})
        if isinstance(row, dict):
            selected[task] = row.get("acc,none")
    selected_by_language = {
        lang: {task: selected.get(task) for task in tasks}
        for lang, tasks in _CANONICAL_RETENTION_LANG_TASKS.items()
    }
    raw_task_count = len(task_results)
    canonical_present = sum(1 for task in canonical if task in task_results)
    return {
        "file": path_info(path),
        "selected_accuracy": selected,
        "selected_accuracy_by_language": selected_by_language,
        "canonical_retention_tasks": canonical,
        "canonical_retention_present_count": canonical_present,
        "task_count": canonical_present,
        "non_canonical_dropped_count": raw_task_count - canonical_present,
    }


def checkpoint_metrics(iter_root, label):
    if not iter_root.exists():
        return {}
    return {
        "native_mcq": native_mcq_metrics(iter_root, label),
        "bpb": bpb_metrics(iter_root),
        "retention": retention_metrics(iter_root),
    }


checkpoint_states = []
for item in checkpoints:
    iter_pad = f"{item['iteration']:07d}"
    checkpoint_dir = train_run_dir / "checkpoints" / f"iter_{iter_pad}"
    iter_eval_root = eval_root / f"iter_{iter_pad}"
    hf_dir = eval_root / f"iter_{iter_pad}_hf"
    checksum_manifest = (
        iter_eval_root
        / "checksums"
        / f"{item['label']}_iter_{iter_pad}_checksum_manifest.json"
    )
    sidecar_manifest = iter_eval_root / "sidecar_jobs.tsv"
    state_file = watch_dir / f"iter_{item['iteration']}.submitted"
    submit_log = watch_dir / f"iter_{item['iteration']}_submit.log"
    manifest_rows = []
    text = read_text(sidecar_manifest)
    if text:
        manifest_rows = list(csv.DictReader(text.splitlines(), delimiter="\t"))
    attempt_history = archived_sidecar_attempts(iter_eval_root)
    checkpoint_states.append(
        {
            **item,
            "iter_pad": iter_pad,
            "checkpoint_dir": str(checkpoint_dir),
            "checkpoint_exists": checkpoint_dir.is_dir(),
            "metadata_exists": (checkpoint_dir / ".metadata").is_file(),
            "hf_dir": str(hf_dir),
            "hf_exists": hf_dir.exists(),
            "checksum_manifest": path_info(checksum_manifest),
            "eval_iter_root": str(iter_eval_root),
            "eval_iter_root_exists": iter_eval_root.exists(),
            "sidecar_manifest": str(sidecar_manifest),
            "sidecar_manifest_exists": sidecar_manifest.is_file(),
            "sidecar_manifest_rows": manifest_rows,
            "sidecar_attempt_history": {
                "archived_attempts": attempt_history,
                "archived_attempt_count": len(attempt_history),
                "archived_attempt_job_ids": [
                    job_id
                    for attempt in attempt_history
                    for job_id in attempt["job_ids"]
                ],
            },
            "metrics": checkpoint_metrics(iter_eval_root, item["label"]),
            "submitted_state_exists": state_file.is_file(),
            "submit_log_exists": submit_log.is_file(),
        }
    )

state = {
    "collected_at_utc": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
    "run_tag": run_tag,
    "canonical_tasks": CANONICAL_TASKS,
    "paths": {
        "run_root": str(run_root),
        "train_run_dir": str(train_run_dir),
        "eval_root": str(eval_root),
        "watch_dir": str(watch_dir),
        "submit_state_dir": str(submit_state_dir),
        "active_log": str(active_log),
    },
    "latest_training": parse_latest_training_line(active_log),
    "queue": queue_rows(),
    "counts": {
        "checkpoint_iter_dirs": sum(1 for p in (train_run_dir / "checkpoints").glob("iter_*") if p.is_dir()),
        "sidecar_files": count_items(watch_dir, "files"),
        "eval_items": count_items(eval_root, "all"),
    },
    "watcher": {
        "env_files": watcher_env_states(watch_dir),
        "stdout_tail": tail_lines(watch_out, 12),
        "stderr": path_info(watch_err),
        "stderr_tail": tail_lines(watch_err, 60) if Path(watch_err).exists() and Path(watch_err).stat().st_size else [],
    },
    "training_chain": parse_training_chain(submit_state_dir / "training_chain.tsv"),
    "checkpoints": checkpoint_states,
    "run_metadata": load_json(train_run_dir / "run_metadata.json"),
    "dataset": {
        "paths_env": str(run_root / "04_hplt_b1_dataset_5b_20260528T112539Z" / "dataset_paths.env"),
        "validation": load_json(run_root / "04_hplt_b1_dataset_5b_20260528T112539Z" / "validation" / "dataset_validation.json"),
        "jsonl_manifest": load_json(run_root / "04_hplt_b1_dataset_5b_20260528T112539Z" / "jsonl" / "hplt_b1_5b.manifest.json"),
        "megatron_manifest": load_json(run_root / "04_hplt_b1_dataset_5b_20260528T112539Z" / "megatron" / "hplt_b1_base_text_document.manifest.json"),
    },
    "preserved_attempts": sorted(
        [
            str(p)
            for p in train_run_dir.iterdir()
            if p.is_dir() and (p.name.startswith("failed_attempt_") or p.name.startswith("canceled_attempt_"))
        ]
    )
    if train_run_dir.exists()
    else [],
}
print(json.dumps(state, ensure_ascii=False, sort_keys=True))
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote-host", default="clariden")
    parser.add_argument("--run-tag", default=DEFAULT_RUN_TAG)
    parser.add_argument("--run-root", default=DEFAULT_RUN_ROOT)
    parser.add_argument("--output", type=Path, help="Optional path for JSON output.")
    parser.add_argument("--compact", action="store_true", help="Emit compact JSON.")
    parser.add_argument(
        "--canonical-tasks",
        type=Path,
        default=DEFAULT_CANONICAL_TASKS,
        help="Canonical task lockdown JSON (goal/canonical_eval_tasks.json).",
    )
    return parser.parse_args()


def run_remote(args: argparse.Namespace) -> Dict[str, Any]:
    canonical_text = args.canonical_tasks.read_text(encoding="utf-8")
    # Parse-validate locally before shipping to the remote so a malformed
    # lockdown file fails fast and visibly.
    json.loads(canonical_text)
    canonical_compact = json.dumps(json.loads(canonical_text), ensure_ascii=False)
    # Quote-safe encoding for shell single-quoted env var assignment.
    canonical_shell = canonical_compact.replace("'", "'\\''")
    env = (
        f"RUN_TAG={args.run_tag!r} RUN_ROOT={args.run_root!r} "
        f"CANONICAL_TASKS_JSON='{canonical_shell}'"
    )
    cmd = [
        "ssh",
        "-o",
        "ConnectTimeout=30",
        args.remote_host,
        f"{env} python3 - <<'PY'\n{REMOTE_PROGRAM}\nPY",
    ]
    proc = subprocess.run(
        cmd,
        universal_newlines=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"remote collector failed with {proc.returncode}:\n{proc.stderr}")
    return json.loads(proc.stdout)


def file_info(path: Path) -> Dict[str, Any]:
    info = {"path": str(path), "exists": path.exists()}
    if path.exists():
        stat = path.stat()
        info["size"] = stat.st_size
        info["mtime"] = (
            dt.datetime.fromtimestamp(stat.st_mtime, dt.timezone.utc)
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z")
        )
    return info


def load_local_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return {"_error": str(exc)}


def safe_label(label: str) -> str:
    return re.sub(r"(^_+|_+$)", "", re.sub(r"[^A-Za-z0-9._-]+", "_", label))


def add_local_review_state(state: Dict[str, Any], subproject_dir: Path) -> None:
    reviews_root = subproject_dir / "adversarial_reviews"
    watch_state = reviews_root / f"{state['run_tag']}_watch_state"
    verify_state = (
        subproject_dir
        / "monitor_logs"
        / f"{state['run_tag']}_sidecar_verify_state"
    )
    verify_watch_log = (
        subproject_dir
        / "monitor_logs"
        / f"{state['run_tag']}_sidecar_verify_watch.log"
    )
    state["local"] = {
        "subproject_dir": str(subproject_dir),
        "review_watch_log": file_info(watch_state / "home_review_watcher.log"),
        "review_watch_done_files": sorted(str(p) for p in watch_state.glob("iter_*.review_done")),
        "sidecar_verify_watch_log": file_info(verify_watch_log),
        "sidecar_verify_done_files": sorted(str(p) for p in verify_state.glob("iter_*.handoff_done")),
        "report_artifacts": {
            "config_geometry_audit_iter_0000119_json": file_info(
                subproject_dir / "reports" / "config_geometry_audit_iter_0000119.json"
            ),
            "config_geometry_audit_iter_0000119_md": file_info(
                subproject_dir / "reports" / "config_geometry_audit_iter_0000119.md"
            ),
            "goal_doc": file_info(subproject_dir / "goal" / "goal.md"),
            "goal_hyperparameters_json": file_info(
                subproject_dir / "goal" / "hyperparameters.json"
            ),
            "run_log_20260528": file_info(subproject_dir / "RUN_LOG_20260528.md"),
            "report_draft_5b": file_info(
                subproject_dir / "reports" / "5B_REPORT_DRAFT.md"
            ),
            "submit_checkpoint_sidecars": file_info(
                subproject_dir / "scripts" / "submit_checkpoint_sidecars.sh"
            ),
            "watch_and_submit_checkpoint_sidecars": file_info(
                subproject_dir
                / "scripts"
                / "watch_and_submit_checkpoint_sidecars.sbatch"
            ),
            "watch_checkpoint_sidecar_verification": file_info(
                subproject_dir / "scripts" / "watch_checkpoint_sidecar_verification.sh"
            ),
            "watch_and_run_adversarial_reviews": file_info(
                subproject_dir / "scripts" / "watch_and_run_adversarial_reviews.sh"
            ),
        },
    }
    for checkpoint in state["checkpoints"]:
        iter_pad = checkpoint["iter_pad"]
        iteration = checkpoint["iteration"]
        latest_verify = (
            subproject_dir
            / "reports"
            / f"iter_{iter_pad}_checkpoint_sidecar_verify_latest.json"
        )
        pass_verify = (
            subproject_dir
            / "reports"
            / f"iter_{iter_pad}_checkpoint_sidecar_handoff_pass.json"
        )
        done_file = verify_state / f"iter_{iteration}.handoff_done"
        latest_verify_json = load_local_json(latest_verify)
        pass_verify_json = load_local_json(pass_verify)
        review_dir = reviews_root / safe_label(checkpoint["label"])
        checkpoint["local_review"] = {
            "dir": str(review_dir),
            "prompt": file_info(review_dir / "prompt.md"),
            "events_jsonl": file_info(review_dir / "codex_events.jsonl"),
            "metadata": file_info(review_dir / "review_metadata.env"),
            "critique": file_info(review_dir / "adversarial_critique.md"),
        }
        checkpoint["local_sidecar_handoff"] = {
            "latest_verify_json": file_info(latest_verify),
            "latest_summary": latest_verify_json.get("summary", {}),
            "pass_verify_json": file_info(pass_verify),
            "pass_summary": pass_verify_json.get("summary", {}),
            "done_file": file_info(done_file),
            "verified": done_file.is_file() and pass_verify.is_file(),
        }


def main() -> int:
    args = parse_args()
    subproject_dir = Path(__file__).resolve().parents[1]
    state = run_remote(args)
    add_local_review_state(state, subproject_dir)
    state["canonical_tasks_source"] = str(args.canonical_tasks)
    state["collector"] = {
        "script": str(Path(__file__).resolve()),
        "host": os.uname().nodename,
    }
    indent = None if args.compact else 2
    text = json.dumps(state, indent=indent, ensure_ascii=False, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
