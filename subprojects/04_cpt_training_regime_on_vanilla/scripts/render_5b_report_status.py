#!/usr/bin/env python3
"""Render a Markdown status block from collect_5b_report_state.py JSON."""

import argparse
import json
from pathlib import Path


def yesno(value):
    return "yes" if value else "no"


def table_row(values):
    return "| " + " | ".join(str(v) for v in values) + " |"


def fmt_float(value, digits=6):
    if isinstance(value, (int, float)):
        return ("{:.%df}" % digits).format(value)
    return ""


def metric_at(mapping, *keys):
    value = mapping
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def next_incomplete_checkpoint(state):
    latest = state.get("latest_training") or {}
    current_iter = latest.get("current_iter")
    try:
        current_iter = int(current_iter)
    except (TypeError, ValueError):
        current_iter = None
    for checkpoint in state.get("checkpoints") or []:
        if not checkpoint.get("metadata_exists"):
            iteration = checkpoint.get("iteration")
            try:
                iteration_int = int(iteration)
            except (TypeError, ValueError):
                iteration_int = None
            remaining = "unknown"
            if current_iter is not None and iteration_int is not None:
                remaining = max(0, iteration_int - current_iter)
            return {
                "label": checkpoint.get("label", ""),
                "iteration": iteration,
                "tokens": checkpoint.get("tokens", ""),
                "remaining_iters": remaining,
            }
    return {}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--state",
        type=Path,
        default=Path("reports/latest_5b_report_state.json"),
        help="JSON state emitted by collect_5b_report_state.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/latest_5b_report_status.md"),
        help="Markdown status output path.",
    )
    return parser.parse_args()


def render_training(state):
    latest = state.get("latest_training", {})
    etas = latest.get("etas") or {}
    next_checkpoint = next_incomplete_checkpoint(state)
    next_iter = str(next_checkpoint.get("iteration", ""))
    lines = [
        "## Latest Training Snapshot",
        "",
        table_row(["Field", "Value"]),
        table_row(["---", "---"]),
        table_row(["Collected at UTC", state.get("collected_at_utc", "unknown")]),
        table_row(["Run tag", state.get("run_tag", "unknown")]),
        table_row(["Current iter", latest.get("current_iter", "unknown")]),
        table_row(["Target iter", latest.get("target_iter", "unknown")]),
        table_row(["Consumed tokens", "{}B".format(latest.get("consumed_tokens_b", "unknown"))]),
        table_row(["LM loss", latest.get("lm_loss", "unknown")]),
        table_row(["Skipped iterations", latest.get("skipped", "unknown")]),
        table_row(["NaN iterations", latest.get("nan", "unknown")]),
        table_row(["Tokens/sec/GPU", latest.get("tokens_sec_gpu", "unknown")]),
        table_row(["Iter ms", latest.get("iter_ms", "unknown")]),
        table_row(["Next incomplete checkpoint", next_checkpoint.get("label", "none") or "none"]),
        table_row(["Next checkpoint iter", next_checkpoint.get("iteration", "none") or "none"]),
        table_row(["Remaining iters to next checkpoint", next_checkpoint.get("remaining_iters", "none")]),
        table_row(["Next checkpoint tokens", next_checkpoint.get("tokens", "none") or "none"]),
        table_row(["Next checkpoint ETA", etas.get(next_iter, "none") if next_iter else "none"]),
        table_row(["ETA iter 119", etas.get("119", "unknown")]),
        table_row(["ETA iter 238", etas.get("238", "unknown")]),
        table_row(["ETA iter 300", etas.get("300", "unknown")]),
        table_row(["ETA iter 477", etas.get("477", "unknown")]),
        table_row(["ETA iter 834", etas.get("834", "unknown")]),
        table_row(["ETA iter 1192", etas.get("1192", "unknown")]),
        "",
        "Latest line:",
        "",
        "```text",
        latest.get("latest_line", ""),
        "```",
        "",
    ]
    return lines


def render_queue(state):
    rows = state.get("queue") or []
    lines = [
        "## Queue",
        "",
        table_row(["Job ID", "Name", "Partition", "State", "Elapsed", "Limit", "Reason"]),
        table_row(["---", "---", "---", "---", "---", "---", "---"]),
    ]
    if rows:
        for row in rows:
            lines.append(
                table_row(
                    [
                        row.get("job_id", ""),
                        row.get("name", ""),
                        row.get("partition", ""),
                        row.get("state", ""),
                        row.get("elapsed", ""),
                        row.get("limit", ""),
                        row.get("reason", ""),
                    ]
                )
            )
    else:
        lines.append(table_row(["none", "", "", "", "", "", ""]))
    lines.append("")
    return lines


def render_counts(state):
    counts = state.get("counts") or {}
    watcher = state.get("watcher") or {}
    stderr = watcher.get("stderr") or {}
    latest = state.get("latest_training") or {}
    lines = [
        "## Artifact Counts And Health",
        "",
        table_row(["Check", "Value"]),
        table_row(["---", "---"]),
        table_row(["Checkpoint iter dirs", counts.get("checkpoint_iter_dirs", "unknown")]),
        table_row(["Sidecar files", counts.get("sidecar_files", "unknown")]),
        table_row(["Eval items", counts.get("eval_items", "unknown")]),
        table_row(["Watcher stderr exists", yesno(stderr.get("exists"))]),
        table_row(["Watcher stderr bytes", stderr.get("size", "unknown")]),
        table_row(["Nonzero skipped/NaN lines", latest.get("nonzero_skipped_or_nan_count", "unknown")]),
        table_row(["Severe log terms", latest.get("severe_term_count", "unknown")]),
        "",
    ]
    return lines


def render_watcher_coverage(state):
    watcher = state.get("watcher") or {}
    env_files = watcher.get("env_files") or []
    lines = [
        "## Sidecar Watcher Coverage",
        "",
        table_row(
            [
                "Watcher job",
                "Started UTC",
                "Max watch seconds",
                "Resubmit if incomplete",
                "Code/math heldouts required",
                "Env mtime",
            ]
        ),
        table_row(["---", "---", "---:", "---", "---", "---"]),
    ]
    if env_files:
        for item in env_files:
            file_info = item.get("file") or {}
            lines.append(
                table_row(
                    [
                        item.get("slurm_job_id", ""),
                        item.get("start_time_utc", ""),
                        item.get("max_watch_seconds", ""),
                        item.get("resubmit_if_incomplete", ""),
                        item.get("require_code_math_heldouts", ""),
                        file_info.get("mtime", ""),
                    ]
                )
            )
    else:
        lines.append(table_row(["none", "", "", "", "", ""]))
    lines.append("")
    return lines


def render_checkpoints(state):
    lines = [
        "## Checkpoints",
        "",
        table_row(
            [
                "Label",
                "Iter",
                "Tokens",
                "Megatron",
                "Metadata",
                "HF",
                "Checksum manifest",
                "Eval root",
                "Sidecar manifest",
                "Archived sidecar attempts",
                "Handoff verified",
                "Review critique",
            ]
        ),
        table_row(["---", "---:", "---:", "---", "---", "---", "---", "---", "---:", "---", "---"]),
    ]
    for checkpoint in state.get("checkpoints") or []:
        review = checkpoint.get("local_review") or {}
        critique = (review.get("critique") or {}).get("exists")
        handoff = checkpoint.get("local_sidecar_handoff") or {}
        attempt_history = checkpoint.get("sidecar_attempt_history") or {}
        lines.append(
            table_row(
                [
                    checkpoint.get("label", ""),
                    checkpoint.get("iteration", ""),
                    checkpoint.get("tokens", ""),
                    yesno(checkpoint.get("checkpoint_exists")),
                    yesno(checkpoint.get("metadata_exists")),
                    yesno(checkpoint.get("hf_exists")),
                    yesno((checkpoint.get("checksum_manifest") or {}).get("exists")),
                    yesno(checkpoint.get("eval_iter_root_exists")),
                    yesno(checkpoint.get("sidecar_manifest_exists")),
                    attempt_history.get("archived_attempt_count", 0),
                    yesno(handoff.get("verified")),
                    yesno(critique),
                ]
            )
        )
    lines.append("")
    return lines


def render_handoff_verification(state):
    lines = [
        "## Handoff Verification",
        "",
        table_row(
            [
                "Label",
                "Iter",
                "Latest check",
                "Checkpoint metadata",
                "Sidecars submitted",
                "Manifest complete",
                "Outputs ready",
                "Active sidecar jobs",
                "Slurm done",
                "Checksum ready",
                "Handoff ready",
                "Pass snapshot",
            ]
        ),
        table_row(["---", "---:", "---", "---", "---", "---", "---", "---:", "---", "---", "---", "---"]),
    ]
    for checkpoint in state.get("checkpoints") or []:
        handoff = checkpoint.get("local_sidecar_handoff") or {}
        latest_file = handoff.get("latest_verify_json") or {}
        latest_summary = handoff.get("latest_summary") or {}
        pass_file = handoff.get("pass_verify_json") or {}
        lines.append(
            table_row(
                [
                    checkpoint.get("label", ""),
                    checkpoint.get("iteration", ""),
                    latest_file.get("mtime", "missing") if latest_file.get("exists") else "missing",
                    yesno(latest_summary.get("checkpoint_metadata_ready")),
                    yesno(latest_summary.get("sidecars_submitted")),
                    yesno(latest_summary.get("manifest_has_all_expected_kinds")),
                    yesno(latest_summary.get("expected_outputs_ready")),
                    latest_summary.get("active_sidecar_jobs", "unknown"),
                    yesno(latest_summary.get("slurm_jobs_completed")),
                    yesno(latest_summary.get("checksum_manifest_ready")),
                    yesno(latest_summary.get("handoff_ready")),
                    yesno(pass_file.get("exists")),
                ]
            )
        )
    lines.append("")
    return lines


def canonical_retention_language_order(state):
    """Order = canonical_eval_tasks.retention.languages, preserving file order."""
    canonical = state.get("canonical_tasks") or {}
    return list((canonical.get("retention") or {}).get("languages") or [])


def canonical_retention_tasks_for_language(state, lang):
    """Tasks per language, dropping entries flagged absent_on_disk."""
    canonical = state.get("canonical_tasks") or {}
    entries = (
        (canonical.get("retention") or {}).get("tasks_per_language", {}).get(lang, [])
    )
    return [entry["task"] for entry in entries if not entry.get("absent_on_disk")]


def render_checkpoint_metrics(state):
    languages = canonical_retention_language_order(state)
    language_columns = [
        (lang, canonical_retention_tasks_for_language(state, lang))
        for lang in languages
    ]
    header_row = (
        [
            "Label",
            "Iter",
            "MCQ headline",
            "MCQ + Plutus",
            "Greek BPB",
            "Greek trunc",
            "Code BPB",
            "Math BPB",
        ]
        + [
            "Retention {} ({})".format(lang, "/".join(tasks)) if tasks else "Retention {} (absent)".format(lang)
            for lang, tasks in language_columns
        ]
    )
    align_row = ["---", "---:", "---:", "---:", "---:", "---:", "---:", "---:"] + [
        "---" for _ in language_columns
    ]
    lines = [
        "## Checkpoint Metrics",
        "",
        table_row(header_row),
        table_row(align_row),
    ]
    any_metrics = False
    for checkpoint in state.get("checkpoints") or []:
        metrics = checkpoint.get("metrics") or {}
        native = metrics.get("native_mcq") or {}
        bpb = metrics.get("bpb") or {}
        retention = metrics.get("retention") or {}
        selected = retention.get("selected_accuracy") or {}
        if not native and not bpb and not selected:
            continue
        any_metrics = True
        greek = bpb.get("greek") or {}
        code = bpb.get("code") or {}
        math = bpb.get("math") or {}
        retention_columns = []
        for _lang, tasks in language_columns:
            if not tasks:
                retention_columns.append("absent")
                continue
            retention_columns.append(
                "/".join(fmt_float(selected.get(task), 3) for task in tasks)
            )
        lines.append(
            table_row(
                [
                    checkpoint.get("label", ""),
                    checkpoint.get("iteration", ""),
                    fmt_float(native.get("headline_macro_accuracy"), 6),
                    fmt_float(native.get("headline_plus_diagnostic_macro_accuracy"), 6),
                    fmt_float(greek.get("bpb"), 6),
                    fmt_float(greek.get("fraction_truncated"), 3),
                    fmt_float(code.get("bpb"), 6),
                    fmt_float(math.get("bpb"), 6),
                ]
                + retention_columns
            )
        )
    if not any_metrics:
        lines.append(table_row(["none"] + [""] * (len(header_row) - 1)))
    lines.append("")
    return lines


def render_report_artifacts(state):
    local = state.get("local") or {}
    artifacts = local.get("report_artifacts") or {}
    lines = [
        "## Report Artifacts",
        "",
        table_row(["Artifact", "Exists", "Updated UTC", "Path"]),
        table_row(["---", "---", "---", "---"]),
    ]
    if artifacts:
        for name, info in sorted(artifacts.items()):
            lines.append(
                table_row(
                    [
                        name,
                        yesno(info.get("exists")),
                        info.get("mtime", ""),
                        "`{}`".format(info.get("path", "")),
                    ]
                )
            )
    else:
        lines.append(table_row(["none", "no", "", ""]))
    lines.append("")
    return lines


def render_paths(state):
    paths = state.get("paths") or {}
    lines = [
        "## Key Paths",
        "",
        table_row(["Name", "Path"]),
        table_row(["---", "---"]),
    ]
    for key in ("train_run_dir", "eval_root", "watch_dir", "submit_state_dir", "active_log"):
        lines.append(table_row([key, "`{}`".format(paths.get(key, ""))]))
    lines.append("")
    return lines


def render_canonical_tasks_header(state):
    canonical = state.get("canonical_tasks") or {}
    source = state.get("canonical_tasks_source", "unknown")
    headline_tasks = (canonical.get("greek_native_mcq") or {}).get("headline_tasks") or []
    diagnostic_tasks = (canonical.get("greek_native_mcq") or {}).get(
        "diagnostic_separate"
    ) or []
    excluded_mt = (canonical.get("greek_native_mcq") or {}).get(
        "excluded_mt_derived"
    ) or []
    retention_languages = (canonical.get("retention") or {}).get("languages") or []
    retention_tasks_per_language = (
        (canonical.get("retention") or {}).get("tasks_per_language") or {}
    )
    retention_present = []
    retention_absent = []
    for lang in retention_languages:
        for entry in retention_tasks_per_language.get(lang, []):
            label = "{}::{}".format(lang, entry.get("task"))
            if entry.get("absent_on_disk"):
                retention_absent.append(label)
            else:
                retention_present.append(label)
    heldout_bpb = canonical.get("heldout_bpb") or []
    lines = [
        "## Canonical Eval Tasks",
        "",
        "Render is filtered against the canonical lockdown file. Non-canonical task data stays on disk; the renderer never surfaces it.",
        "",
        table_row(["Field", "Value"]),
        table_row(["---", "---"]),
        table_row(["canonical_tasks_source", "`{}`".format(source)]),
        table_row(["schema", canonical.get("schema", "unknown")]),
        table_row(["generated_utc", canonical.get("generated_utc", "unknown")]),
        table_row(["greek_native_mcq.headline_tasks", ", ".join(headline_tasks) or "none"]),
        table_row(["greek_native_mcq.diagnostic_separate", ", ".join(diagnostic_tasks) or "none"]),
        table_row(
            ["greek_native_mcq.excluded_mt_derived", ", ".join(excluded_mt) or "none"]
        ),
        table_row(["retention.languages", ", ".join(retention_languages) or "none"]),
        table_row(
            [
                "retention.tasks_present",
                ", ".join(retention_present) or "none",
            ]
        ),
        table_row(
            [
                "retention.tasks_absent_on_disk",
                ", ".join(retention_absent) or "none",
            ]
        ),
        table_row(["heldout_bpb", ", ".join(heldout_bpb) or "none"]),
        "",
    ]
    return lines


def main():
    args = parse_args()
    state = json.loads(args.state.read_text(encoding="utf-8"))
    lines = [
        "# Latest 5B Report Status",
        "",
        "Generated from `{}`.".format(args.state),
        "",
        "Canonical task lockdown: `{}`.".format(
            state.get("canonical_tasks_source", "unknown")
        ),
        "",
    ]
    for section in (
        render_canonical_tasks_header,
        render_training,
        render_counts,
        render_queue,
        render_watcher_coverage,
        render_checkpoints,
        render_handoff_verification,
        render_checkpoint_metrics,
        render_report_artifacts,
        render_paths,
    ):
        lines.extend(section(state))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
