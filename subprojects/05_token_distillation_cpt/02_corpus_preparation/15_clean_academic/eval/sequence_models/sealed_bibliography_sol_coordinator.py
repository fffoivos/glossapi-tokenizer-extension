#!/usr/bin/env python3
"""Stream sealed Clariden role batches through local schema-bound Codex calls.

The coordinator never downloads a packet or document.  It requests at most
two chunks over SSH, keeps that bounded envelope in memory, runs an ephemeral
read-only Codex call using an explicitly selected model and reasoning effort,
and streams the JSON response back to Clariden for semantic validation and
immutable storage.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


ALLOWED_MODELS = ("gpt-5.6-sol", "gpt-5.6-terra")
ALLOWED_EFFORTS = ("low", "medium", "high", "xhigh", "max", "ultra")
REMOTE_MODULE = "sequence_models.sealed_bibliography_test"
QUALITY_DOCUMENT_MAX_CHARS = 100_000
QUALITY_BATCH_MAX_CHARS = 180_000


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _remote_command(args: argparse.Namespace, arguments: Sequence[str], stdin: str | None = None) -> str:
    command = [
        "uenv",
        "run",
        args.remote_uenv,
        "--view=default",
        "--",
        "env",
        f"PYTHONPATH={args.remote_pythonpath}",
        args.remote_python,
        "-m",
        REMOTE_MODULE,
        *arguments,
    ]
    completed = subprocess.run(
        ["ssh", args.ssh_host, shlex.join(command)],
        input=stdin,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=args.ssh_timeout_seconds,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"remote command failed ({completed.returncode}): {completed.stderr[-3000:]}"
        )
    return completed.stdout


def _json_output(text: str, operation: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"{operation} did not return one JSON object: {text[-1000:]}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{operation} returned a non-object")
    return value


def _codex(
    prompt: str,
    schema: Path,
    envelope: Mapping[str, Any],
    timeout: int,
    model: str,
    reasoning_effort: str,
) -> dict[str, Any]:
    full_prompt = prompt.rstrip() + "\n\n" + json.dumps(envelope, ensure_ascii=False) + "\n"
    with tempfile.TemporaryDirectory(prefix="sealed-bib-codex-") as directory:
        workspace = Path(directory) / "empty-read-only-workspace"
        workspace.mkdir()
        response_path = Path(directory) / "response.json"
        command = [
            "codex",
            "exec",
            "--model",
            model,
            "--sandbox",
            "read-only",
            "--ephemeral",
            "--skip-git-repo-check",
            "--cd",
            str(workspace),
            "--config",
            f'model_reasoning_effort="{reasoning_effort}"',
            "--output-schema",
            str(schema),
            "--output-last-message",
            str(response_path),
            "-",
        ]
        completed = subprocess.run(
            command,
            input=full_prompt,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"Codex exited {completed.returncode}: {completed.stderr[-3000:]}"
            )
        if not response_path.is_file():
            raise RuntimeError("Codex produced no schema-bound final response")
        value = json.loads(response_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError("Codex final response is not a JSON object")
    return value


def _run_one(
    args: argparse.Namespace,
    *,
    index: int,
    prompt: str,
    schema: Path,
) -> dict[str, Any]:
    export_command = "export-batch" if args.kind == "role" else "export-quality-batch"
    ingest_command = "ingest-batch" if args.kind == "role" else "ingest-quality-batch"
    export = _json_output(
        _remote_command(
            args,
            [
                export_command,
                "--packet",
                args.remote_packet,
                "--run-dir",
                args.remote_run_dir,
                "--batch-index",
                str(index),
            ],
        ),
        f"export batch {index}",
    )
    if export.get("status") == "complete":
        return {"batch_index": index, "status": "already_complete", "batch_id": export["batch_id"]}
    if export.get("status") != "pending":
        raise RuntimeError(f"batch {index}: unexpected remote status {export.get('status')!r}")
    response = _codex(
        prompt,
        schema,
        export,
        args.codex_timeout_seconds,
        args.model,
        args.reasoning_effort,
    )
    accepted = _json_output(
        _remote_command(
            args,
            [
                ingest_command,
                "--packet",
                args.remote_packet,
                "--run-dir",
                args.remote_run_dir,
                "--batch-index",
                str(index),
            ],
            stdin=json.dumps(response, ensure_ascii=False),
        ),
        f"ingest batch {index}",
    )
    if accepted.get("status") != "accepted":
        raise RuntimeError(f"batch {index}: remote validator did not accept the response")
    return {"batch_index": index, "status": "accepted", "batch_id": accepted["batch_id"]}


def run(args: argparse.Namespace) -> int:
    role_passes = {"pass-a", "pass-b", "adjudication"}
    quality_passes = {"quality-a", "quality-b", "quality-c"}
    if (args.kind == "role" and args.pass_id not in role_passes) or (
        args.kind == "quality" and args.pass_id not in quality_passes
    ):
        raise ValueError(f"pass-id {args.pass_id!r} is incompatible with kind {args.kind!r}")
    prompt_path = Path(args.local_prompt).resolve()
    schema_path = Path(args.local_output_schema).resolve()
    prompt = prompt_path.read_text(encoding="utf-8")
    prepare_command = "prepare-run" if args.kind == "role" else "prepare-quality-run"
    prepare_arguments = [
        prepare_command,
        "--packet",
        args.remote_packet,
        "--pass-id",
        args.pass_id,
        "--reviewer-id",
        args.reviewer_id,
        "--model",
        args.model,
        "--reasoning-effort",
        args.reasoning_effort,
        "--prompt",
        args.remote_prompt,
        "--output-schema",
        args.remote_output_schema,
        "--batch-size",
        str(args.batch_size),
        "--run-dir",
        args.remote_run_dir,
    ]
    if args.kind == "quality":
        prepare_arguments.extend(
            [
                "--max-document-characters",
                str(args.quality_document_max_characters),
                "--max-batch-characters",
                str(args.quality_batch_max_characters),
            ]
        )
    contract = _json_output(
        _remote_command(args, prepare_arguments),
        "prepare run",
    )
    if contract.get("model") != args.model or contract.get("reasoning_effort") != args.reasoning_effort:
        raise RuntimeError("remote contract differs from the requested model/reasoning effort")
    if contract.get("prompt_sha256") != _sha256(prompt_path):
        raise RuntimeError("local and remote prompts differ")
    if contract.get("output_schema_sha256") != _sha256(schema_path):
        raise RuntimeError("local and remote output schemas differ")
    if args.kind == "quality":
        caps = contract.get("quality_character_caps", {})
        if (
            int(caps.get("serialized_per_document", -1))
            != args.quality_document_max_characters
            or int(caps.get("serialized_per_batch", -1))
            != args.quality_batch_max_characters
        ):
            raise RuntimeError("remote quality character caps differ from the local request")
    total_count = int(contract["batch_count"])
    if not 0 <= args.start_batch_index < total_count:
        raise ValueError("start-batch-index is outside the run contract")
    stop_index = total_count
    if args.maximum_batches is not None:
        if args.maximum_batches <= 0:
            raise ValueError("maximum-batches must be positive")
        stop_index = min(total_count, args.start_batch_index + args.maximum_batches)
    batch_indices = range(args.start_batch_index, stop_index)
    results: list[dict[str, Any]] = []
    failures: list[tuple[int, BaseException]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_index = {
            executor.submit(
                _run_one, args, index=index, prompt=prompt, schema=schema_path
            ): index
            for index in batch_indices
        }
        for future in concurrent.futures.as_completed(future_to_index):
            index = future_to_index[future]
            try:
                result = future.result()
                results.append(result)
                print(json.dumps(result, sort_keys=True), flush=True)
            except BaseException as error:
                failures.append((index, error))
    if failures:
        detail = "; ".join(
            f"batch {index}: {type(error).__name__}: {error}"
            for index, error in sorted(failures)
        )
        raise RuntimeError(
            f"{len(failures)} batches failed; successful batches are immutable and a rerun resumes: {detail}"
        )
    if args.maximum_batches is None:
        finalize_command = "finalize-pass" if args.kind == "role" else "finalize-quality-pass"
        final = _json_output(
            _remote_command(
                args,
                [
                    finalize_command,
                    "--packet",
                    args.remote_packet,
                    "--run-dir",
                    args.remote_run_dir,
                    "--output",
                    args.remote_pass_output,
                ],
            ),
            "finalize pass",
        )
        print(
            json.dumps(
                {
                    "status": "pass_complete",
                    "kind": args.kind,
                    "item_count": final.get("line_count", final.get("document_count")),
                    "overlap_exact_role_agreement": final.get("overlap_exact_role_agreement"),
                },
                sort_keys=True,
            )
        )
    else:
        print(
            json.dumps(
                {
                    "status": "bounded_preflight_complete",
                    "start_batch_index": args.start_batch_index,
                    "batches": len(batch_indices),
                },
                sort_keys=True,
            )
        )
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("role", "quality"), default="role")
    parser.add_argument("--ssh-host", default="clariden")
    parser.add_argument("--remote-python", default="python3")
    parser.add_argument("--remote-uenv", default="pytorch/v2.9.1:v2")
    parser.add_argument("--remote-pythonpath", required=True)
    parser.add_argument("--remote-packet", required=True)
    parser.add_argument("--remote-run-dir", required=True)
    parser.add_argument("--remote-pass-output", required=True)
    parser.add_argument(
        "--pass-id",
        choices=("pass-a", "pass-b", "adjudication", "quality-a", "quality-b", "quality-c"),
        required=True,
    )
    parser.add_argument("--reviewer-id", required=True)
    parser.add_argument("--model", choices=ALLOWED_MODELS, required=True)
    parser.add_argument(
        "--reasoning-effort", choices=ALLOWED_EFFORTS, default="medium"
    )
    parser.add_argument("--local-prompt", required=True)
    parser.add_argument("--remote-prompt", required=True)
    parser.add_argument("--local-output-schema", required=True)
    parser.add_argument("--remote-output-schema", required=True)
    parser.add_argument("--batch-size", type=int, choices=(1, 2), default=2)
    parser.add_argument(
        "--quality-document-max-characters", type=int, default=QUALITY_DOCUMENT_MAX_CHARS
    )
    parser.add_argument(
        "--quality-batch-max-characters", type=int, default=QUALITY_BATCH_MAX_CHARS
    )
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    parser.add_argument("--maximum-batches", type=int)
    parser.add_argument("--start-batch-index", type=int, default=0)
    parser.add_argument("--ssh-timeout-seconds", type=int, default=120)
    parser.add_argument("--codex-timeout-seconds", type=int, default=1800)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
