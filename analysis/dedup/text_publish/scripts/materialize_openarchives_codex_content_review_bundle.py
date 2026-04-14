from __future__ import annotations

import argparse
import json
import shlex
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


DEFAULT_CODEX_MODEL = "gpt-5.4-mini"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _utc_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _safe_slug(value: str) -> str:
    out = []
    for char in str(value or ""):
        if char.isalnum() or char in {"-", "_"}:
            out.append(char.lower())
        else:
            out.append("-")
    return "".join(out).strip("-")[:48] or "job"


def build_prompt(packet: dict[str, Any]) -> str:
    instructions = [
        "You are reviewing one OpenArchives exact-duplicate packet.",
        "Your task is only to classify what the shared text is at a high level.",
        "Do not try to decide which metadata row owns the text in this step.",
        "Classify the shared text into one of these labels:",
        "- repository_notice",
        "- historical_review_bibliography",
        "- full_thesis_or_article",
        "- technical_manual",
        "- formula_junk",
        "- image_markup_artifact",
        "- other",
        "Also decide whether the shared text is real substantive content or problematic output.",
        "Ground the decision in the text itself with 1-3 short matched signals.",
    ]
    return (
        "\n".join(instructions)
        + "\n\nPACKET_METADATA\n"
        + json.dumps(
            {
                "packet_family_id": packet["packet_family_id"],
                "group_hash": packet["group_hash"],
                "group_size": packet["group_size"],
                "packet_mode": packet["packet_mode"],
                "batch_index": packet["batch_index"],
                "batch_count": packet["batch_count"],
                "collections_in_group": packet["collections_in_group"],
                "suspicious_subreason": packet["suspicious_subreason"],
                "deterministic_resolution_status": packet["deterministic_resolution_status"],
                "content_size_hint": packet["content_size_hint"],
                "shared_text_profile": packet["shared_text_profile"],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n\nROWS\n"
        + json.dumps(packet["rows"], ensure_ascii=False, indent=2)
        + "\n\nSHARED_TEXT\n"
        + str(packet["shared_text"])
        + "\n"
    )


def build_output_schema(packet: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "packet_family_id": {"type": "string", "enum": [str(packet["packet_family_id"])]},
            "group_hash": {"type": "string", "enum": [str(packet["group_hash"])]},
            "packet_mode": {"type": "string", "enum": [str(packet["packet_mode"])]},
            "batch_index": {"type": "integer", "enum": [int(packet["batch_index"])]},
            "high_level_content_type": {
                "type": "string",
                "enum": [
                    "repository_notice",
                    "historical_review_bibliography",
                    "full_thesis_or_article",
                    "technical_manual",
                    "formula_junk",
                    "image_markup_artifact",
                    "other",
                ],
            },
            "is_substantive_content": {"type": "boolean"},
            "supports_gemini_owner_resolution": {"type": "boolean"},
            "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
            "reasoning_summary": {"type": "string"},
            "matched_signals": {"type": "array", "items": {"type": "string"}},
        },
        "required": [
            "packet_family_id",
            "group_hash",
            "packet_mode",
            "batch_index",
            "high_level_content_type",
            "is_substantive_content",
            "supports_gemini_owner_resolution",
            "confidence",
            "reasoning_summary",
            "matched_signals",
        ],
    }


def build_codex_command(*, prompt_path: Path, schema_path: Path, output_path: Path, model: str, project_root: Path) -> list[str]:
    return [
        "codex",
        "exec",
        "--skip-git-repo-check",
        "--cd",
        str(project_root),
        "--sandbox",
        "workspace-write",
        "--model",
        model,
        "--output-schema",
        str(schema_path),
        "--output-last-message",
        str(output_path),
        "-",
    ]


def runner_script_lines(*, model: str) -> list[str]:
    return [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f'MODEL="${{MODEL:-{model}}}"',
        'PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
        'for job_dir in "$PROJECT_ROOT"/jobs/*; do',
        '  [ -d "$job_dir" ] || continue',
        '  if [ -f "$job_dir/response.json" ]; then',
        '    echo "skip $(basename "$job_dir") response.json already exists"',
        "    continue",
        "  fi",
        '  echo "run $(basename "$job_dir")"',
        '  codex exec --skip-git-repo-check --cd "$PROJECT_ROOT" --sandbox workspace-write --model "$MODEL" --output-schema "$job_dir/output_schema.json" --output-last-message "$job_dir/response.json" - < "$job_dir/prompt.txt"',
        "done",
    ]


def materialize_openarchives_codex_content_review_bundle(
    run_root: Path,
    *,
    bundle_name: str | None = None,
    model: str = DEFAULT_CODEX_MODEL,
    overwrite: bool = False,
) -> dict[str, Any]:
    run_root = run_root.resolve()
    analysis_root = run_root / "analysis"
    packet_index_path = analysis_root / "oa_gemini_resolution_packets" / "index.json"
    packet_index = load_json(packet_index_path)
    if not packet_index:
        raise ValueError(f"no packets found in {packet_index_path}")

    bundle_id = bundle_name or f"oa_codex_content_review_{_utc_stamp()}"
    bundle_dir = analysis_root / bundle_id
    if bundle_dir.exists() and not overwrite:
        raise FileExistsError(f"bundle already exists: {bundle_dir}")
    bundle_dir.mkdir(parents=True, exist_ok=True)
    jobs_dir = bundle_dir / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, Any]] = []
    for entry in packet_index:
        packet_path = Path(entry["packet_path"])
        packet = load_json(packet_path)
        slug = f"{int(entry['packet_id']):04d}_{_safe_slug(str(packet['group_hash'])[:16])}_b{int(entry['batch_index']):02d}"
        job_dir = jobs_dir / slug
        job_dir.mkdir(parents=True, exist_ok=True)

        prompt_path = job_dir / "prompt.txt"
        schema_path = job_dir / "output_schema.json"
        source_path = job_dir / "source.json"
        copied_packet_path = job_dir / "packet.json"
        response_path = job_dir / "response.json"

        prompt_path.write_text(build_prompt(packet) + "\n", encoding="utf-8")
        schema_path.write_text(json.dumps(build_output_schema(packet), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        source_path.write_text(json.dumps(entry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        copied_packet_path.write_text(json.dumps(packet, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        command = build_codex_command(
            prompt_path=prompt_path,
            schema_path=schema_path,
            output_path=response_path,
            model=model,
            project_root=bundle_dir,
        )
        manifest_rows.append(
            {
                "packet_id": int(entry["packet_id"]),
                "group_hash": str(entry["group_hash"]),
                "packet_mode": str(entry["packet_mode"]),
                "batch_index": int(entry["batch_index"]),
                "batch_count": int(entry["batch_count"]),
                "job_dir": str(job_dir),
                "packet_path": str(packet_path),
                "command_preview": shlex.join(command) + f" < {shlex.quote(str(prompt_path))}",
            }
        )

    runner_path = bundle_dir / "run_codex_jobs.sh"
    runner_path.write_text("\n".join(runner_script_lines(model=model)) + "\n", encoding="utf-8")
    runner_path.chmod(0o755)

    jobs_jsonl_path = bundle_dir / "jobs.jsonl"
    jobs_jsonl_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in manifest_rows), encoding="utf-8")

    bundle_manifest = {
        "ok": True,
        "action": "materialize_codex_content_review_bundle",
        "bundle_id": bundle_id,
        "bundle_dir": str(bundle_dir),
        "runner_path": str(runner_path),
        "jobs_jsonl_path": str(jobs_jsonl_path),
        "packet_index_path": str(packet_index_path),
        "job_count": len(manifest_rows),
        "model": model,
        "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    }
    (bundle_dir / "bundle_manifest.json").write_text(json.dumps(bundle_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return bundle_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize a Codex-auth content-review bundle for OA semantic packets.")
    parser.add_argument("--run-root", required=True, type=Path, help="Path to completed exact-stage run root")
    parser.add_argument("--bundle-name", type=str, default=None, help="Optional stable bundle directory name")
    parser.add_argument("--model", type=str, default=DEFAULT_CODEX_MODEL, help="Codex model name for the runner script")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the bundle directory if it already exists")
    args = parser.parse_args()
    payload = materialize_openarchives_codex_content_review_bundle(
        run_root=args.run_root,
        bundle_name=args.bundle_name,
        model=args.model,
        overwrite=args.overwrite,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
