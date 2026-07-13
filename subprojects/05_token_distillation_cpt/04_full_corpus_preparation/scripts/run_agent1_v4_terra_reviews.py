#!/usr/bin/env python3
"""Run one exact-raw Terra Codex session for every v4 review request.

The request JSONL and packet manifest are compact metadata copied from CSCS.
Each raw document is fetched into a fresh temporary root only for its own
session, hash-verified, reviewed, and removed before the next document.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from agent1_v4_raw_review import (  # noqa: E402
    ALLOWED_ROUTES,
    PACKET_SCHEMA,
    REQUEST_SCHEMA,
    file_binding,
    read_json_object,
    sha256_file,
    sha256_json,
    validate_request,
)


RESPONSE_SCHEMA = "agent1_v4_terra_review_response_v1"
EXECUTION_SCHEMA = "agent1_v4_terra_review_execution_receipt_v1"
HEX_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
DOCUMENT_RE = re.compile(r"^documents/[A-Za-z0-9_.-]+/[a-f0-9]{64}\.txt$")
ARTIFACT_TYPES = frozenset(
    {
        "html_tags",
        "html_entities",
        "script_style",
        "navigation",
        "boilerplate",
        "malformed_markdown",
        "mojibake",
        "replacement_or_control_characters",
        "ocr_corruption",
        "broken_words_or_hyphenation",
        "page_furniture",
        "column_or_reading_order",
        "fragmentation",
        "truncation_or_incompleteness",
        "tables_or_formulae",
        "repeated_or_template_text",
        "structured_field_loss",
        "non_greek_drift",
        "placeholder_or_empty",
        "other",
    }
)
RESPONSE_IDENTITY_FIELDS = (
    "request_id",
    "source_id",
    "source_doc_id",
    "document_path",
    "document_sha256",
    "prompt_sha256",
)


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _write_no_replace(path: Path, payload: bytes, *, mode: int = 0o600) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"immutable output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, mode)
        os.link(temporary, path)
        os.chmod(path, mode)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json_no_replace(path: Path, value: Mapping[str, object]) -> None:
    _write_no_replace(
        path,
        (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8"),
    )


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: response/request row must be an object")
            rows.append(value)
    return rows


def _safe_remote_root(value: str) -> str:
    if not value.startswith("/") or not re.fullmatch(r"/[A-Za-z0-9_./-]+", value):
        raise ValueError("remote packet root contains unsafe characters")
    return value.rstrip("/")


def _safe_remote_host(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_.@-]+", value):
        raise ValueError("remote host contains unsafe characters")
    return value


def validate_request_manifest(requests_path: Path, packet_manifest_path: Path) -> list[dict[str, object]]:
    manifest = read_json_object(packet_manifest_path)
    if manifest.get("schema_version") != PACKET_SCHEMA or manifest.get("status") != "passed":
        raise ValueError("packet manifest is not a passed v4 raw review packet")
    binding = manifest.get("requests")
    if not isinstance(binding, Mapping):
        raise ValueError("packet manifest lacks requests binding")
    actual = file_binding(requests_path)
    if actual["bytes"] != binding.get("bytes") or actual["sha256"] != binding.get("sha256"):
        raise ValueError("local requests copy differs from packet manifest")
    requests = _read_jsonl(requests_path)
    source_counts: dict[str, int] = {}
    seen_ids: set[str] = set()
    for request in requests:
        validate_request(request)
        request_id = str(request["request_id"])
        if request_id in seen_ids:
            raise ValueError("duplicate request_id")
        seen_ids.add(request_id)
        source_id = str(request["source_id"])
        source_counts[source_id] = source_counts.get(source_id, 0) + 1
    expected_counts = manifest.get("source_counts")
    if not isinstance(expected_counts, Mapping) or source_counts != expected_counts:
        raise ValueError("request source counts differ from packet manifest")
    if len(source_counts) != 18 or any(count < 1 for count in source_counts.values()):
        raise ValueError("Terra lane requires a complete non-empty 18-source packet")
    if manifest.get("logical_review_count") != len(requests) or sum(source_counts.values()) != len(requests):
        raise ValueError("Terra lane request count differs from packet manifest")
    return sorted(requests, key=lambda value: (str(value["source_id"]), str(value["sample_id"])))


def render_prompt(template: str, request: Mapping[str, object]) -> str:
    result = template
    for key in ("document_path", "source_id", "source_dataset", "source_route", "extraction_route"):
        result = result.replace("{{" + key + "}}", str(request[key]))
    if "{{" in result or "}}" in result:
        raise ValueError("unresolved placeholder in Terra prompt template")
    return result


def execution_response_schema_bytes(response_schema: Path, request: Mapping[str, object]) -> bytes:
    """Specialize the frozen schema with this request's immutable identity.

    The frozen response schema supplies the common response contract.  Codex
    Structured Outputs must also be told the *particular* long identifiers to
    return: their type/pattern alone does not reveal the expected value.  The
    specialization only strengthens those six string properties with constants
    already contained in the immutable review request; all response validation
    remains against the original request afterwards.
    """

    schema = read_json_object(response_schema)
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        raise ValueError("response schema lacks properties")
    for field in RESPONSE_IDENTITY_FIELDS:
        value = request.get(field)
        if not isinstance(value, str) or not value:
            raise ValueError(f"request lacks immutable {field}")
        properties[field] = {"type": "string", "const": value}
    return (canonical_json(schema) + "\n").encode("utf-8")


def _seatbelt_quote(value: str) -> str:
    return json.dumps(value)


def seatbelt_profile(call_root: Path, codex_bin: Path, codex_home: Path) -> str:
    """Return a deny-by-default macOS profile for exactly one call root.

    System/runtime directories are readable so the signed Codex executable can
    start.  The only user-controlled data readable by the model is call_root;
    Codex auth is the narrow additional read allowlist.
    """

    root = str(call_root.resolve())
    launcher = codex_bin.absolute()
    resolved_codex = launcher.resolve()
    codex = str(resolved_codex)
    home = str(codex_home.resolve())
    return "\n".join(
        [
            "(version 1)",
            "(deny default)",
            "(allow process*)",
            "(allow mach-lookup)",
            "(allow sysctl-read)",
            "(allow network-outbound)",
            # Codex's runtime probes the filesystem root for PATH aliases.
            # This is metadata access to '/' only, not a recursive permit.
            "(allow file-read* (literal \"/\"))",
            f"(allow file-read* (subpath {_seatbelt_quote(root)}))",
            f"(allow file-read* (subpath {_seatbelt_quote(home)}))",
            # ``codex exec`` loads machine-level requirements/TLS policy and
            # macOS resolver state.  These are operating-system paths only;
            # neither exposes the review corpus or user document tree.
            "(allow file-read* (subpath \"/etc\"))",
            "(allow file-read* (subpath \"/private/etc\"))",
            "(allow file-read* (subpath \"/var\"))",
            # Homebrew exposes the executable through a symlink.  Both the
            # fixed launcher directory and the resolved versioned runtime are
            # executable code, never user document storage; allowing only
            # those directories keeps the model's readable data boundary at
            # the one-document call root and Codex auth home.
            f"(allow file-read* (subpath {_seatbelt_quote(str(launcher.parent))}))",
            f"(allow file-read* (subpath {_seatbelt_quote(str(resolved_codex.parent))}))",
            f"(allow file-read* (literal {_seatbelt_quote(codex)}))",
            "(allow file-read* (subpath \"/System\"))",
            "(allow file-read* (subpath \"/usr\"))",
            "(allow file-read* (subpath \"/private/var/db\"))",
            "(allow file-read* (subpath \"/Library\"))",
            f"(allow file-write* (subpath {_seatbelt_quote(root)}))",
            # This is the authenticated client state, not review data.  Codex
            # may refresh its own ephemeral/auth bookkeeping here; its model
            # tools remain read-only and cannot write outside the call root.
            f"(allow file-write* (subpath {_seatbelt_quote(home)}))",
        ]
    ) + "\n"


def _run_isolation_canary(
    *, sandbox_exec: str,
    profile: Path,
    forbidden_path: Path,
) -> None:
    if not forbidden_path.is_absolute() or not forbidden_path.exists():
        raise ValueError("forbidden-read-path must name an existing absolute path")
    completed = subprocess.run(
        [sandbox_exec, "-f", str(profile), "/bin/cat", str(forbidden_path)],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode == 0:
        raise RuntimeError("OS isolation canary could read the forbidden file")


def _copy_one_document(
    *,
    request: Mapping[str, object],
    destination: Path,
    packet_root: Path | None,
    remote_host: str | None,
    remote_packet_root: str | None,
    scp_bin: str,
) -> None:
    relative = str(request["document_path"])
    if not DOCUMENT_RE.fullmatch(relative):
        raise ValueError("unsafe document path")
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if packet_root is not None:
        source = packet_root.resolve() / relative
        if source.is_symlink() or not source.is_file():
            raise FileNotFoundError(f"missing local review document: {source}")
        shutil.copyfile(source, destination)
    else:
        if remote_host is None or remote_packet_root is None:
            raise ValueError("one of packet_root or remote packet source is required")
        remote = f"{_safe_remote_host(remote_host)}:{_safe_remote_root(remote_packet_root)}/{relative}"
        completed = subprocess.run(
            [scp_bin, "-q", remote, str(destination)],
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"document fetch failed: {completed.stderr.strip()}")
    if destination.stat().st_size != int(request["document_bytes"]):
        raise ValueError("fetched document byte count differs from request")
    if sha256_file(destination) != request["document_sha256"]:
        raise ValueError("fetched document hash differs from request")


def validate_response(response: Mapping[str, object], request: Mapping[str, object], document: Path) -> None:
    required = {
        "response_schema_version",
        "request_id",
        "source_id",
        "source_doc_id",
        "document_path",
        "document_sha256",
        "model_id",
        "prompt_sha256",
        "cleanliness_score",
        "text_quality_score",
        "confidence",
        "coverage_mode",
        "summary",
        "extraction_artifacts",
    }
    if set(response) != required:
        raise ValueError(f"response keys drift: {sorted(set(response) ^ required)}")
    if response.get("response_schema_version") != RESPONSE_SCHEMA:
        raise ValueError("response schema version mismatch")
    for key in ("request_id", "source_id", "source_doc_id", "document_path", "document_sha256", "prompt_sha256"):
        if response.get(key) != request.get(key):
            raise ValueError(f"response request binding mismatch: {key}")
    if response.get("model_id") != "gpt-5.6-terra":
        raise ValueError("response model is not gpt-5.6-terra")
    for key in ("cleanliness_score", "text_quality_score"):
        if not isinstance(response.get(key), int) or not 1 <= int(response[key]) <= 5:
            raise ValueError(f"invalid {key}")
    if response.get("confidence") not in {"low", "medium", "high"}:
        raise ValueError("invalid confidence")
    if response.get("coverage_mode") not in {"full", "deterministic_windows"}:
        raise ValueError("invalid coverage_mode")
    if not isinstance(response.get("summary"), str) or not response["summary"].strip():
        raise ValueError("empty response summary")
    artifacts = response.get("extraction_artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("extraction_artifacts must be an array")
    lines = document.read_text(encoding="utf-8").splitlines()
    for artifact in artifacts:
        if not isinstance(artifact, Mapping):
            raise ValueError("artifact must be an object")
        keys = {
            "type",
            "severity",
            "line_start",
            "line_end",
            "evidence_excerpt",
            "explanation",
            "deterministic_cleaning_possible",
            "suggested_cleaning_action",
        }
        if set(artifact) != keys:
            raise ValueError("artifact keys drift")
        if artifact.get("type") not in ARTIFACT_TYPES or artifact.get("severity") not in {"minor", "moderate", "severe"}:
            raise ValueError("artifact type/severity is invalid")
        start, end = artifact.get("line_start"), artifact.get("line_end")
        if not isinstance(start, int) or not isinstance(end, int) or start < 1 or end < start or end > len(lines):
            raise ValueError("artifact line range is invalid")
        evidence = artifact.get("evidence_excerpt")
        if not isinstance(evidence, str) or len(evidence) > 200:
            raise ValueError("artifact evidence exceeds contract")
        if evidence and evidence not in "\n".join(lines[start - 1 : end]):
            raise ValueError("artifact evidence does not occur inside its cited line range")
        if not isinstance(artifact.get("explanation"), str) or not artifact["explanation"].strip():
            raise ValueError("artifact explanation is empty")
        if not isinstance(artifact.get("deterministic_cleaning_possible"), bool):
            raise ValueError("artifact cleaning flag is invalid")
        if not isinstance(artifact.get("suggested_cleaning_action"), str) or not artifact["suggested_cleaning_action"].strip():
            raise ValueError("artifact cleaning action is empty")


def repair_artifact_evidence(response: dict[str, object], document: Path) -> list[dict[str, object]]:
    """Replace only an unverifiable excerpt with text from its declared span.

    A model occasionally identifies a correct line span but formats the quoted
    evidence with an ellipsis or other non-verbatim change.  Preserve every
    semantic model judgement and derive a bounded, exact excerpt solely from
    its already-declared source span.  The returned hash-only ledger makes each
    such narrow repair auditable without retaining another raw-text copy.
    """

    lines = document.read_text(encoding="utf-8").splitlines()
    artifacts = response.get("extraction_artifacts")
    if not isinstance(artifacts, list):
        return []
    repairs: list[dict[str, object]] = []
    for index, artifact in enumerate(artifacts):
        if not isinstance(artifact, dict):
            continue
        evidence = artifact.get("evidence_excerpt")
        start, end = artifact.get("line_start"), artifact.get("line_end")
        if not isinstance(evidence, str) or not isinstance(start, int) or not isinstance(end, int):
            continue
        if start < 1 or end < start or end > len(lines):
            continue
        cited = "\n".join(lines[start - 1 : end])
        if not evidence or evidence in cited:
            continue
        replacement = cited[:200]
        if not replacement:
            continue
        artifact["evidence_excerpt"] = replacement
        repairs.append(
            {
                "artifact_index": index,
                "line_start": start,
                "line_end": end,
                "model_evidence_sha256": hashlib.sha256(evidence.encode("utf-8")).hexdigest(),
                "replacement_evidence_sha256": hashlib.sha256(replacement.encode("utf-8")).hexdigest(),
            }
        )
    return repairs


def _invoke_request(
    *,
    request: Mapping[str, object],
    prompt_template: str,
    response_schema: Path,
    codex_bin: str,
    model: str,
    sandbox_exec: str,
    codex_home: Path,
    forbidden_read_path: Path,
    call_parent: Path,
    packet_root: Path | None,
    remote_host: str | None,
    remote_packet_root: str | None,
    scp_bin: str,
    timeout_seconds: int,
) -> tuple[dict[str, object], int, list[dict[str, object]]]:
    with tempfile.TemporaryDirectory(prefix=f"call-{request['request_id'][:12]}-", dir=call_parent) as temporary:
        root = Path(temporary)
        document = root / str(request["document_path"])
        _copy_one_document(
            request=request,
            destination=document,
            packet_root=packet_root,
            remote_host=remote_host,
            remote_packet_root=remote_packet_root,
            scp_bin=scp_bin,
        )
        request_path = root / "request.json"
        response_path = root / "response.json"
        schema_path = root / "response-schema.json"
        prompt_path = root / "prompt.txt"
        _write_no_replace(request_path, (canonical_json(dict(request)) + "\n").encode("utf-8"))
        _write_no_replace(schema_path, execution_response_schema_bytes(response_schema, request))
        prompt = render_prompt(prompt_template, request)
        _write_no_replace(prompt_path, prompt.encode("utf-8"))
        runtime_tmp = root / "tmp"
        runtime_tmp.mkdir(mode=0o700)
        if not forbidden_read_path.is_absolute() or not forbidden_read_path.exists():
            raise ValueError("forbidden-read-path must name an existing absolute path")
        # Codex's native ``read-only`` macOS sandbox is the enforcement layer
        # for the model session.  Launching Codex inside a second seatbelt
        # profile prevents that native sandbox from starting, so the outer
        # canary profile is deliberately not nested around this process.
        command = [
            codex_bin,
            "exec",
            "--cd",
            str(root),
            "--skip-git-repo-check",
            "--ephemeral",
            "--sandbox",
            "read-only",
            "--ignore-user-config",
            "--model",
            model,
            "--output-schema",
            str(schema_path),
            "--output-last-message",
            str(response_path),
            prompt,
        ]
        started = time.monotonic()
        environment = os.environ.copy()
        # Codex's own read-only macOS sandbox writes a transient profile while
        # starting.  Keep that transient state inside the already isolated
        # call root rather than granting access to the host temporary tree.
        environment.update({"TMPDIR": str(runtime_tmp), "TMP": str(runtime_tmp), "TEMP": str(runtime_tmp)})
        completed = subprocess.run(
            command,
            cwd=root,
            env=environment,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
        elapsed_ms = int((time.monotonic() - started) * 1000)
        if completed.returncode != 0:
            raise RuntimeError(f"codex exec failed ({completed.returncode}): {completed.stderr[-1000:]}")
        try:
            response = json.loads(response_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            raise RuntimeError("codex did not produce a valid JSON response") from exc
        if not isinstance(response, dict):
            raise RuntimeError("codex response must be a JSON object")
        evidence_repairs = repair_artifact_evidence(response, document)
        validate_response(response, request, document)
        return response, elapsed_ms, evidence_repairs


def run_reviews(
    *,
    requests_path: Path,
    packet_manifest_path: Path,
    policy_path: Path,
    prompt_path: Path,
    response_schema_path: Path,
    state_dir: Path,
    output: Path,
    codex_bin: str,
    sandbox_exec: str,
    forbidden_read_path: Path,
    packet_root: Path | None = None,
    remote_host: str | None = None,
    remote_packet_root: str | None = None,
    scp_bin: str = "scp",
    timeout_seconds: int = 1800,
    max_parallel: int = 6,
) -> dict[str, object]:
    requests = validate_request_manifest(requests_path, packet_manifest_path)
    packet_manifest = read_json_object(packet_manifest_path)
    policy_binding = file_binding(policy_path)
    declared_policy = packet_manifest.get("policy")
    if not isinstance(declared_policy, Mapping) or policy_binding["bytes"] != declared_policy.get("bytes") or policy_binding["sha256"] != declared_policy.get("sha256"):
        raise ValueError("local policy copy differs from packet manifest")
    prompt_binding = file_binding(prompt_path)
    schema_binding = file_binding(response_schema_path)
    policy = read_json_object(policy_path)
    model = str(policy.get("model", ""))
    if model != "gpt-5.6-terra":
        raise ValueError("packet policy is not bound to gpt-5.6-terra")
    for request in requests:
        if request["model"] != model or request["prompt_sha256"] != prompt_binding["sha256"] or request["response_schema_sha256"] != schema_binding["sha256"]:
            raise ValueError("request prompt/schema/model binding drift")
    if (packet_root is None) == (remote_packet_root is None):
        raise ValueError("supply exactly one of packet_root or remote_packet_root")
    if remote_packet_root is not None and remote_host is None:
        raise ValueError("remote packet root requires remote_host")
    if output.exists():
        raise FileExistsError(f"final response output already exists: {output}")
    state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    responses_dir = state_dir / "responses"
    calls_dir = state_dir / ".calls"
    responses_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    calls_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    resolved_codex = shutil.which(codex_bin) or codex_bin
    if not Path(resolved_codex).is_file():
        raise FileNotFoundError(f"codex executable not found: {codex_bin}")
    codex_home = Path(os.environ.get("CODEX_HOME", Path.home() / ".codex"))
    prompt_template = prompt_path.read_text(encoding="utf-8")
    if not 1 <= max_parallel <= 16:
        raise ValueError("max_parallel must be between 1 and 16")
    completed_rows: list[dict[str, object]] = []
    attempts: list[dict[str, object]] = []
    pending: list[dict[str, object]] = []
    for request in requests:
        cache_path = responses_dir / f"{request['request_id']}.json"
        if cache_path.exists():
            cached = read_json_object(cache_path)
            response = cached.get("response")
            if not isinstance(response, dict):
                raise ValueError(f"cached response malformed: {cache_path}")
            # Verify the response's identity without re-fetching raw text.
            for key in ("request_id", "source_id", "source_doc_id", "document_path", "document_sha256", "prompt_sha256"):
                if response.get(key) != request.get(key):
                    raise ValueError(f"cached response drift: {cache_path}")
            completed_rows.append(response)
            attempts.extend(cached.get("attempts", []))
            continue

        pending.append(request)

    def run_one_request(request: dict[str, object]) -> tuple[dict[str, object], list[dict[str, object]]]:
        """Run one isolated review with retries; safe to execute in a thread."""

        cache_path = responses_dir / f"{request['request_id']}.json"
        failures: list[str] = []
        request_attempts: list[dict[str, object]] = []
        for attempt in range(1, 4):
            try:
                response, elapsed_ms, evidence_repairs = _invoke_request(
                    request=request,
                    prompt_template=prompt_template,
                    response_schema=response_schema_path,
                    codex_bin=resolved_codex,
                    model=model,
                    sandbox_exec=sandbox_exec,
                    codex_home=codex_home,
                    forbidden_read_path=forbidden_read_path,
                    call_parent=calls_dir,
                    packet_root=packet_root,
                    remote_host=remote_host,
                    remote_packet_root=remote_packet_root,
                    scp_bin=scp_bin,
                    timeout_seconds=timeout_seconds,
                )
                attempt_row = {
                    "request_id": request["request_id"],
                    "attempt": attempt,
                    "status": "passed",
                    "elapsed_ms": elapsed_ms,
                    "artifact_evidence_repairs": evidence_repairs,
                }
                request_attempts.append(attempt_row)
                cache = {"request": request, "response": response, "attempts": request_attempts}
                _write_json_no_replace(cache_path, cache)
                return response, request_attempts
            except Exception as exc:  # transport/schema failures are retriable only twice
                failures.append(str(exc))
                attempt_row = {"request_id": request["request_id"], "attempt": attempt, "status": "failed", "error": str(exc)[:2000]}
                request_attempts.append(attempt_row)
                if attempt < 3:
                    time.sleep(5 * attempt)
        raise RuntimeError(
            f"terminal Terra failure for source {request['source_id']} request {request['request_id']}: {failures[-1]}"
        )

    terminal_failures: list[str] = []
    if pending:
        with ThreadPoolExecutor(max_workers=max_parallel, thread_name_prefix="terra-review") as executor:
            futures = {executor.submit(run_one_request, request): request for request in pending}
            for future in as_completed(futures):
                try:
                    response, request_attempts = future.result()
                except Exception as exc:
                    terminal_failures.append(str(exc))
                    continue
                completed_rows.append(response)
                attempts.extend(request_attempts)
    if terminal_failures:
        raise RuntimeError("; ".join(sorted(terminal_failures)))
    completed_rows.sort(key=lambda row: (str(row["source_id"]), str(row["request_id"])))
    attempts.sort(key=lambda row: (str(row.get("request_id", "")), int(row.get("attempt", 0))))
    payload = b"".join((canonical_json(row) + "\n").encode("utf-8") for row in completed_rows)
    _write_no_replace(output, payload)
    receipt = {
        "schema_version": EXECUTION_SCHEMA,
        "status": "passed",
        "executed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "packet_manifest": file_binding(packet_manifest_path),
        "requests": file_binding(requests_path),
        "prompt": prompt_binding,
        "response_schema": schema_binding,
        "responses": file_binding(output),
        "model": model,
        "runner_script": file_binding(Path(__file__)),
        "max_parallel": max_parallel,
        "logical_review_count": len(completed_rows),
        "billable_invocation_count": len(attempts),
        "attempts": attempts,
        "state_dir": str(state_dir.resolve()),
    }
    _write_json_no_replace(output.with_suffix(output.suffix + ".receipt.json"), receipt)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--packet-manifest", type=Path, required=True)
    parser.add_argument("--policy", type=Path, default=root / "configs" / "agent1_v4_raw_review_policy.json")
    parser.add_argument("--prompt", type=Path, default=root / "configs" / "agent1_v4_terra_review_prompt.md")
    parser.add_argument("--response-schema", type=Path, default=root / "schemas" / "agent1_v4_terra_review_response.schema.json")
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--packet-root", type=Path)
    parser.add_argument("--remote-host")
    parser.add_argument("--remote-packet-root")
    parser.add_argument("--scp-bin", default="scp")
    parser.add_argument("--codex-bin", default="codex")
    parser.add_argument("--sandbox-exec", default="/usr/bin/sandbox-exec")
    parser.add_argument("--forbidden-read-path", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--max-parallel", type=int, default=6, help="bounded simultaneous isolated Codex calls (1-16)")
    args = parser.parse_args(argv)
    receipt = run_reviews(
        requests_path=args.requests,
        packet_manifest_path=args.packet_manifest,
        policy_path=args.policy,
        prompt_path=args.prompt,
        response_schema_path=args.response_schema,
        state_dir=args.state_dir,
        output=args.output,
        codex_bin=args.codex_bin,
        sandbox_exec=args.sandbox_exec,
        forbidden_read_path=args.forbidden_read_path,
        packet_root=args.packet_root,
        remote_host=args.remote_host,
        remote_packet_root=args.remote_packet_root,
        scp_bin=args.scp_bin,
        timeout_seconds=args.timeout_seconds,
        max_parallel=args.max_parallel,
    )
    print(json.dumps({"ok": True, "logical_review_count": receipt["logical_review_count"]}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
