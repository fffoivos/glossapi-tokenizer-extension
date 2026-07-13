from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parents[1]
SCRIPT = HERE / "clariden" / "agent1_v3_pre_review.sh"


def write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def option(arguments: list[str], name: str) -> str:
    return arguments[arguments.index(name) + 1]


def make_clean_git_repo(root: Path) -> str:
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.invalid"], check=True)
    subprocess.run(["git", "-C", str(root), "config", "user.name", "Test"], check=True)
    (root / "README").write_text("clean\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(root), "add", "README"], check=True)
    subprocess.run(["git", "-C", str(root), "commit", "-qm", "initial"], check=True)
    return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()


def make_contract_script(path: Path) -> None:
    write_executable(
        path,
        """#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

args = sys.argv[1:]
command = args[0]
def one(name):
    return args[args.index(name) + 1]
def many(name):
    result = []
    index = 0
    while True:
        try:
            index = args.index(name, index)
        except ValueError:
            return result
        result.append(args[index + 1])
        index += 2

root = Path(one('--run-root'))
stage = one('--stage')
data_root = Path(one('--data-root')) if '--data-root' in args else Path(os.environ['AGENT1_V3_DATA_ROOT'])
if command == 'get-stage-output':
    outputs = json.loads((root / 'stages' / stage / 'outputs.json').read_text(encoding='utf-8'))
    print(outputs[one('--basename')])
elif command == 'get-stage-attempt-dir':
    outputs = json.loads((root / 'stages' / stage / 'outputs.json').read_text(encoding='utf-8'))
    attempt_id = Path(next(iter(outputs.values()))).parent.name
    if one('--storage') == 'metadata':
        print(root / 'stages' / stage / 'attempts' / attempt_id)
    else:
        print(data_root / 'stages' / stage / 'attempts' / attempt_id)
else:
    attempt_id = one('--attempt-id')
    attempt = root / 'stages' / stage / 'attempts' / attempt_id
    data_attempt = data_root / 'stages' / stage / 'attempts' / attempt_id
if command == 'begin-stage':
    attempt.mkdir(parents=True)
    data_attempt.mkdir(parents=True)
    (root / 'stages' / stage / 'stage_contract.json').write_text(
        json.dumps({'inputs': args, 'stage': stage, 'attempt_id': attempt_id}), encoding='utf-8'
    )
    print(json.dumps({'ok': True, 'attempt_dir': str(attempt), 'data_attempt_dir': str(data_attempt)}))
elif command == 'finish-stage':
    outputs = [Path(value) for value in many('--output')]
    assert outputs and all(
        path.is_file() and (attempt in path.parents or data_attempt in path.parents)
        for path in outputs
    )
    stage_root = root / 'stages' / stage
    (stage_root / 'outputs.json').write_text(
        json.dumps({path.name: str(path) for path in outputs}, sort_keys=True), encoding='utf-8'
    )
    (stage_root / 'stage_receipt.json').write_text('{}', encoding='utf-8')
    print(json.dumps({'ok': True}))
elif command not in {'get-stage-output', 'get-stage-attempt-dir'}:
    raise SystemExit(command)
""",
    )


def make_fake_scripts(phase: Path) -> None:
    scripts = phase / "scripts"
    scripts.mkdir(parents=True)
    write_executable(
        scripts / "normalize_sources.py",
        """#!/usr/bin/env python3
import hashlib
import json
import sys
from pathlib import Path

args = sys.argv[1:]
def opt(name): return Path(args[args.index(name) + 1])
output, manifest = opt('--output'), opt('--manifest')
roster_path = opt('--candidate-roster').resolve()
roster = json.loads(roster_path.read_text(encoding='utf-8'))
candidates = roster['candidate_source_ids']
review_routes = roster['review_routes']
source_routes = roster.get('source_routes', review_routes)
extraction_routes = roster.get('extraction_routes', review_routes)
route_declarations = {
    source: {
        'source_route': source_routes[source],
        'review_route': review_routes[source],
        'extraction_route': extraction_routes[source],
    }
    for source in candidates
}
candidate_roster = {
    'path': str(roster_path),
    'bytes': roster_path.stat().st_size,
    'sha256': hashlib.sha256(roster_path.read_bytes()).hexdigest(),
    'schema_version': roster['schema_version'],
    'base_source_id': roster['base_source_id'],
    'candidate_source_ids': candidates,
    'review_routes': review_routes,
    'source_routes': source_routes,
    'extraction_routes': extraction_routes,
    'route_declarations': route_declarations,
}
for source in ('nanochat_base', 'source-a'):
    shard = output / source / 'part.parquet'
    shard.parent.mkdir(parents=True, exist_ok=True)
    shard.write_text('parquet-placeholder', encoding='utf-8')
manifest.write_text(json.dumps({
    'schema_version': 'full_cpt_normalization_manifest_v1',
    'candidate_roster': candidate_roster,
    'candidate_roster_source_coverage': {
        'status': 'passed',
        'candidate_source_ids': candidates,
        'normalizable_registry_source_ids': ['nanochat_base', *candidates],
        'acquisition_artifact_source_ids': ['nanochat_base', *candidates],
    },
    'candidate_roster_canonical_route_coverage': {
        'schema_version': 'agent1_v3_canonical_route_coverage_v1',
        'status': 'passed',
        'sources': [
            {
                'source_id': source,
                **route_declarations[source],
                'normalized_documents': 1,
                'status': 'passed',
            }
            for source in candidates
        ],
    },
    'sources': [
        {'source_id': 'nanochat_base', 'counts': {'documents_emitted': 1}},
        {'source_id': 'source-a', 'counts': {'documents_emitted': 1}},
    ],
}), encoding='utf-8')
""",
    )
    write_executable(
        scripts / "build_source_lineage.py",
        """#!/usr/bin/env python3
import sys
from pathlib import Path

args = sys.argv[1:]
def opt(name): return Path(args[args.index(name) + 1])
for flag in ('--registry-manifest-out', '--actions-out', '--novelty-out', '--summary-out'):
    path = opt(flag)
    path.write_text('{}' if path.suffix == '.json' else '{}\\n', encoding='utf-8')
""",
    )
    write_executable(
        scripts / "agent1_v3_review.py",
        """#!/usr/bin/env python3
import hashlib
import json
import os
import sys
from pathlib import Path

args = sys.argv[1:]
assert args[0] == 'validate-roster'
output = Path(args[args.index('--output') + 1])
roster_path = Path(args[args.index('--roster') + 1])
roster = json.loads(roster_path.read_text(encoding='utf-8'))
candidates = roster['candidate_source_ids']
review_routes = roster['review_routes']
source_routes = roster.get('source_routes', review_routes)
extraction_routes = roster.get('extraction_routes', review_routes)
output.write_text(json.dumps({
    'schema_version': 'agent1_v3_candidate_roster_route_validation_v1',
    'roster_sha256': hashlib.sha256(roster_path.read_bytes()).hexdigest(),
    'candidate_count': len(candidates),
    'candidate_source_ids': sorted(candidates),
    'logical_source_priority': 'logical_source_then_observed_extraction',
    'source_routes': {source: source_routes[source] for source in sorted(candidates)},
    'review_routes': {source: review_routes[source] for source in sorted(candidates)},
    'extraction_routes': {source: extraction_routes[source] for source in sorted(candidates)},
    'allowed_observed_extraction_routes': {
        source: sorted({source_routes[source], extraction_routes[source]})
        for source in sorted(candidates)
    },
    'inventory_only_exclusion_count': 0,
}), encoding='utf-8')
with Path(os.environ['FAKE_LOG']).open('a', encoding='utf-8') as handle:
    handle.write('validate-roster\\n')
""",
    )
    write_executable(
        scripts / "profile_dataset_quality_rust.py",
        """#!/usr/bin/env python3
import hashlib
import json
import os
import sys
from pathlib import Path

args = sys.argv[1:]
log = Path(os.environ['FAKE_LOG'])
if args[0] == 'validate-build-receipt':
    log.write_text(log.read_text(encoding='utf-8') + 'validate-build\\n', encoding='utf-8')
elif args[0] == 'run':
    def opt(name): return args[args.index(name) + 1]
    import pyarrow as pa
    import pyarrow.parquet as pq
    root = Path(opt('--output-dir'))
    root.mkdir(parents=True)
    documents = root / 'dataset_quality_document_v1.parquet'
    source_ids = [args[index + 1] for index, value in enumerate(args) if value == '--source-id']
    source_route = os.environ.get('FAKE_QUALITY_SOURCE_ROUTE', 'pdf_ocr')
    review_route = os.environ.get('FAKE_QUALITY_REVIEW_ROUTE', source_route)
    extraction_route = os.environ.get('FAKE_QUALITY_EXTRACTION_ROUTE', source_route)
    observed_route = os.environ.get('FAKE_QUALITY_OBSERVED_ROUTE', extraction_route)
    observed_basis = 'explicit_row_route'
    observed_evidence = 'raw_field:format'
    observed_priority = os.environ.get(
        'FAKE_QUALITY_OBSERVED_PRIORITY',
        'logical_primary' if observed_route == source_route else 'secondary_exception_only',
    )
    pq.write_table(pa.table({
        'schema_version': ['dataset_quality_document_v1'] * len(source_ids),
        'source_id': source_ids,
        'source_route': [source_route] * len(source_ids),
        'review_route': [review_route] * len(source_ids),
        'extraction_route': [extraction_route] * len(source_ids),
        'observed_extraction_route': [observed_route] * len(source_ids),
        'observed_extraction_route_basis': [observed_basis] * len(source_ids),
        'observed_extraction_route_evidence': [observed_evidence] * len(source_ids),
        'observed_extraction_route_priority': [observed_priority] * len(source_ids),
    }), documents)
    document_bytes = documents.stat().st_size
    document_sha256 = hashlib.sha256(documents.read_bytes()).hexdigest()
    route_tuples = [
        {
            'source_id': source_id,
            'source_route': source_route,
            'review_route': review_route,
            'extraction_route': extraction_route,
            'observed_extraction_route': observed_route,
            'observed_extraction_route_basis': observed_basis,
            'observed_extraction_route_evidence': observed_evidence,
            'observed_extraction_route_priority': observed_priority,
            'documents': 1,
        }
        for source_id in sorted(source_ids)
    ]
    route_coverage = {
        'schema_version': 'dataset_quality_route_coverage_v1',
        'documents': len(source_ids),
        'source_route_counts': [{'route': source_route, 'documents': len(source_ids)}],
        'review_route_counts': [{'route': review_route, 'documents': len(source_ids)}],
        'extraction_route_counts': [{'route': extraction_route, 'documents': len(source_ids)}],
        'observed_extraction_route_counts': [{'route': observed_route, 'documents': len(source_ids)}],
        'observed_extraction_route_basis_counts': [{'basis': observed_basis, 'documents': len(source_ids)}],
        'observed_extraction_route_priority_counts': [{'priority': observed_priority, 'documents': len(source_ids)}],
        'sources': [
            {
                'source_id': source_id,
                'documents': 1,
                'source_route': source_route,
                'review_route': review_route,
                'extraction_route': extraction_route,
                'observed_extraction_route_counts': [{'route': observed_route, 'documents': 1}],
                'observed_extraction_route_basis_counts': [{'basis': observed_basis, 'documents': 1}],
                'observed_extraction_route_priority_counts': [{'priority': observed_priority, 'documents': 1}],
            }
            for source_id in sorted(source_ids)
        ],
        'route_tuples': route_tuples,
    }
    contract = root / 'contract.json'
    contract.write_text('{}', encoding='utf-8')
    contract_bytes = contract.stat().st_size
    contract_sha256 = hashlib.sha256(contract.read_bytes()).hexdigest()
    (root / 'dataset_quality_summary_v1.json').write_text(json.dumps({
        'schema_version': 'dataset_quality_summary_v1',
        'status': 'passed',
        'scan_mode': 'full_scan',
        'selected_source_ids': source_ids,
        'contract': {
            'path': contract.name,
            'bytes': contract_bytes,
            'sha256': contract_sha256,
        },
        'document_output': {
            'path': documents.name,
            'bytes': document_bytes,
            'sha256': document_sha256,
            'rows': len(source_ids),
        },
        'route_coverage': route_coverage,
    }), encoding='utf-8')
    Path(opt('--site-handoff')).write_text(json.dumps({
        'schema_version': 'dataset_quality_site_handoff_v1',
        'status': 'passed',
        'scan_mode': 'full_scan',
        'route_coverage': route_coverage,
    }), encoding='utf-8')
    log.write_text(log.read_text(encoding='utf-8') + 'full-scan\\n', encoding='utf-8')
else:
    raise SystemExit(args)
""",
    )
    write_executable(
        scripts / "agent1_v3_review_packet.py",
        """#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

args = sys.argv[1:]
def opt(name): return Path(args[args.index(name) + 1])
opt('--output').write_text('{"request": true}\\n', encoding='utf-8')
opt('--manifest').write_text('{"manifest": true}\\n', encoding='utf-8')
Path(os.environ['FAKE_PACKET_ARGS']).write_text(json.dumps(args), encoding='utf-8')
log = Path(os.environ['FAKE_LOG'])
log.write_text(log.read_text(encoding='utf-8') + 'packet\\n', encoding='utf-8')
""",
    )


def make_fake_commands(root: Path) -> Path:
    commands = root / "bin"
    commands.mkdir()
    write_executable(
        commands / "uenv",
        """#!/usr/bin/env bash
set -euo pipefail
[[ $1 == run ]]
shift 2
[[ $1 == --view=default ]]
shift
[[ $1 == -- ]]
shift
if [[ ${2:-} == -c ]]; then
    exit 0
fi
exec "$@"
""",
    )
    write_executable(
        commands / "scontrol",
        """#!/usr/bin/env bash
echo 'JobId=1 ReqTRES=cpu=4,mem=4G AllocTRES=cpu=4'
""",
    )
    return commands


def fixture_environment(tmp_path: Path) -> tuple[dict[str, str], Path, Path]:
    import pyarrow

    phase = tmp_path / "phase"
    make_fake_scripts(phase)
    commands = make_fake_commands(tmp_path)
    contract = tmp_path / "contract.py"
    make_contract_script(contract)
    repo = tmp_path / "repo"
    commit = make_clean_git_repo(repo)
    runtime = tmp_path / "runtime"
    (runtime / "bin").mkdir(parents=True)
    os.symlink(sys.executable, runtime / "bin" / "python")
    config = tmp_path / "config"
    config.mkdir()
    roster = config / "roster.json"
    write_json(
        roster,
        {
            "schema_version": "agent1_full_corpus_v3_candidate_roster_v1",
            "base_source_id": "nanochat_base",
            "candidate_source_ids": ["source-a"],
            "review_routes": {"source-a": "pdf_ocr"},
        },
    )
    policy = config / "policy.json"
    write_json(
        policy,
        {
            "schema_version": "agent1_full_corpus_v3_policy_v1",
            "review": {
                "seed": "frozen-seed",
                "required_model": "gpt-5.6-luna",
                "model_environment_variable": "CODEX_REVIEW_MODEL",
                "no_model_fallback": True,
            },
        },
    )
    for name in ("sources.json", "aliases.json", "nanochat.json"):
        write_json(config / name, {})
    prompt = config / "prompt.md"
    prompt.write_text("prompt", encoding="utf-8")
    schema = config / "schema.json"
    schema.write_text("{}", encoding="utf-8")
    run_id = "agent1-full-corpus-v3-20260713T120000Z-abcdef1"
    runs_root = tmp_path / "runs"
    run_root = runs_root / run_id
    merged = run_root / "phase0" / "merged_acquisition_receipt.json"
    merged.parent.mkdir(parents=True)
    merged.write_text("{}", encoding="utf-8")
    quality_commit = "a" * 40
    data_root = tmp_path / "data" / run_id
    runtime_root = data_root / "runtime" / f"glossapi-rust-quality-{quality_commit}"
    build_receipt = runtime_root / "build_receipt.json"
    build_receipt.parent.mkdir(parents=True)
    build_receipt.write_text("{}", encoding="utf-8")
    modules = build_receipt.parent / "modules"
    modules.mkdir()
    write_json(
        run_root / "run_contract.json",
        {
            "inputs": {
                "glossapi_build_receipt": {
                    "path": str(build_receipt.resolve()),
                    "bytes": build_receipt.stat().st_size,
                    "sha256": hashlib.sha256(build_receipt.read_bytes()).hexdigest(),
                }
            }
        },
    )
    log = tmp_path / "log.txt"
    log.write_text("", encoding="utf-8")
    packet_args = tmp_path / "packet-args.json"
    pyarrow_site_packages = str(Path(pyarrow.__file__).resolve().parents[1])
    inherited_pythonpath = os.environ.get("PYTHONPATH", "")
    environment = {
        **os.environ,
        "PATH": f"{commands}:{os.environ['PATH']}",
        "PYTHONPATH": ":".join(
            part for part in (pyarrow_site_packages, inherited_pythonpath) if part
        ),
        "REPO_ROOT": str(repo),
        "PHASE04_DIR": str(phase),
        "AGENT1_V3_CONTRACT_SCRIPT": str(contract),
        "AGENT1_V3_RUNS_ROOT": str(runs_root),
        "AGENT1_V3_DATA_ROOT_BASE": str(tmp_path / "data"),
        "AGENT1_V3_RUN_ID": run_id,
        "AGENT1_V3_RUNTIME_VENV": str(runtime),
        "AGENT1_V3_SOURCE_CONFIG": str(config / "sources.json"),
        "AGENT1_V3_SOURCE_ALIASES": str(config / "aliases.json"),
        "AGENT1_V3_CANDIDATE_ROSTER": str(roster),
        "AGENT1_V3_POLICY": str(policy),
        "AGENT1_V3_REVIEW_POLICY": str(policy),
        "AGENT1_V3_REVIEW_PROMPT": str(prompt),
        "AGENT1_V3_REVIEW_RESPONSE_SCHEMA": str(schema),
        "AGENT1_V3_NANOCHAT_INITIAL_ROSTER": str(config / "nanochat.json"),
        "AGENT1_V3_GLOSSAPI_BUILD_RECEIPT": str(build_receipt),
        "AGENT1_V3_GLOSSAPI_MODULE_DIR": str(modules),
        "AGENT1_V3_GLOSSAPI_COMMIT": quality_commit,
        "AGENT1_V3_EXPECTED_COMMIT": commit,
        "SLURM_JOB_PARTITION": "normal",
        "SLURM_CPUS_PER_TASK": "4",
        "CODEX_REVIEW_MODEL": "gpt-5.6-luna",
        "FAKE_LOG": str(log),
        "FAKE_PACKET_ARGS": str(packet_args),
    }
    return environment, run_root, packet_args


def run_action(environment: dict[str, str], action: str, job_id: int) -> None:
    subprocess.run(
        ["bash", str(SCRIPT), action],
        check=True,
        text=True,
        capture_output=True,
        env={**environment, "SLURM_JOB_ID": str(job_id)},
    )


def test_review_packet_rejects_build_receipt_drift_from_frozen_contract(tmp_path: Path) -> None:
    environment, run_root, _ = fixture_environment(tmp_path)
    run_action(environment, "normalize", 101)
    run_action(environment, "lineage", 102)
    Path(environment["AGENT1_V3_GLOSSAPI_BUILD_RECEIPT"]).write_text(
        '{"tampered": true}', encoding="utf-8"
    )

    result = subprocess.run(
        ["bash", str(SCRIPT), "review-packet"],
        text=True,
        capture_output=True,
        env={**environment, "SLURM_JOB_ID": "103"},
    )

    assert result.returncode != 0
    assert "differs from the v3 frozen run contract" in result.stderr
    assert not (run_root / "stages" / "30-review-packet").exists()


def test_pre_review_executor_runs_cpu_stages_and_full_scan_before_packet(tmp_path: Path) -> None:
    environment, run_root, packet_args_path = fixture_environment(tmp_path)
    run_action(environment, "normalize", 101)
    run_action(environment, "lineage", 102)
    run_action(environment, "review-packet", 103)

    normalize_attempt = run_root / "stages" / "10-normalize" / "attempts" / "101"
    lineage_attempt = run_root / "stages" / "20-lineage" / "attempts" / "102"
    review_attempt = run_root / "stages" / "30-review-packet" / "attempts" / "103"
    data_root = Path(environment["AGENT1_V3_DATA_ROOT_BASE"]) / environment["AGENT1_V3_RUN_ID"]
    normalize_data_attempt = data_root / "stages" / "10-normalize" / "attempts" / "101"
    lineage_data_attempt = data_root / "stages" / "20-lineage" / "attempts" / "102"
    review_data_attempt = data_root / "stages" / "30-review-packet" / "attempts" / "103"
    coverage = json.loads((normalize_attempt / "normalization_roster_coverage.json").read_text())
    assert coverage["review_routes"] == {"source-a": "pdf_ocr"}
    assert (normalize_data_attempt / "canonical" / "source-a" / "part.parquet").is_file()
    assert not (normalize_attempt / "canonical").exists()
    assert (lineage_attempt / "summary.json").is_file()
    assert (lineage_data_attempt / "document_actions.jsonl").is_file()
    assert (review_attempt / "full_scan_evidence_validation.json").is_file()
    full_scan_validation = json.loads(
        (review_attempt / "full_scan_evidence_validation.json").read_text()
    )
    assert full_scan_validation["logical_source_priority"] == "logical_source_then_observed_extraction"
    assert full_scan_validation["source_routes"] == {"source-a": "pdf_ocr"}
    assert full_scan_validation["extraction_routes"] == {"source-a": "pdf_ocr"}
    assert full_scan_validation["route_coverage_validated_from_document_parquet"] is True
    assert full_scan_validation["source_route_coverage"] == [
        {
            "source_id": "source-a",
            "documents": 1,
            "source_route": "pdf_ocr",
            "review_route": "pdf_ocr",
            "extraction_route": "pdf_ocr",
            "observed_extraction_route_counts": [
                {"route": "pdf_ocr", "documents": 1}
            ],
            "observed_extraction_route_basis_counts": [
                {"basis": "explicit_row_route", "documents": 1}
            ],
            "observed_extraction_route_priority_counts": [
                {"priority": "logical_primary", "documents": 1}
            ],
        }
    ]
    assert full_scan_validation["observed_extraction_route_counts"] == [
        {"route": "pdf_ocr", "documents": 1}
    ]
    assert full_scan_validation["observed_extraction_route_priority_counts"] == [
        {"priority": "logical_primary", "documents": 1}
    ]
    assert (review_attempt / "review_requests.jsonl").is_file()
    assert (review_attempt / "review_packet_manifest.json").is_file()
    assert (review_data_attempt / "quality-full-scan" / "contract.json").is_file()
    assert (review_data_attempt / "quality-full-scan" / "dataset_quality_document_v1.parquet").is_file()
    assert not (review_attempt / "quality-full-scan").exists()

    packet_args = json.loads(packet_args_path.read_text())
    assert option(packet_args, "--prompt") == environment["AGENT1_V3_REVIEW_PROMPT"]
    assert option(packet_args, "--response-schema") == environment["AGENT1_V3_REVIEW_RESPONSE_SCHEMA"]
    assert option(packet_args, "--roster") == environment["AGENT1_V3_CANDIDATE_ROSTER"]
    assert option(packet_args, "--model") == "gpt-5.6-luna"
    assert option(packet_args, "--seed") == "frozen-seed"
    assert option(packet_args, "--full-scan-evidence") == str(
        review_data_attempt / "quality-full-scan" / "dataset_quality_document_v1.parquet"
    )
    assert (Path(environment["FAKE_LOG"]).read_text().splitlines()) == [
        "validate-roster",
        "validate-build",
        "full-scan",
        "packet",
    ]

    for stage, attempt, data_attempt in (
        ("10-normalize", normalize_attempt, normalize_data_attempt),
        ("20-lineage", lineage_attempt, lineage_data_attempt),
        ("30-review-packet", review_attempt, review_data_attempt),
    ):
        outputs = json.loads((run_root / "stages" / stage / "outputs.json").read_text())
        assert outputs
        assert all(
            Path(path).is_relative_to(attempt) or Path(path).is_relative_to(data_attempt)
            for path in outputs.values()
        )


def test_pre_review_script_uses_uenv_for_stage_contract_and_has_no_model_invocation() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert 'path=$(run_python "$AGENT1_V3_CONTRACT_SCRIPT" get-stage-output' in source
    assert "get-stage-attempt-dir" in source
    assert '"$AGENT1_V3_DATA_ATTEMPT_DIR/canonical"' in source
    assert '"$AGENT1_V3_DATA_ATTEMPT_DIR/quality-full-scan"' in source
    assert "--scan-mode full_scan" in source
    assert '"${quality_source_args[@]}"' in source
    assert '"$AGENT1_V3_REVIEW_PROMPT"' in source
    assert '"$AGENT1_V3_REVIEW_RESPONSE_SCHEMA"' in source
    assert "agent1_v3_begin_stage" in source
    assert "agent1_v3_finish_stage" in source
    assert "codex exec" not in source


def test_pre_review_rejects_document_route_drift_even_when_static_roster_is_valid(
    tmp_path: Path,
) -> None:
    environment, _, _ = fixture_environment(tmp_path)
    run_action(environment, "normalize", 101)
    run_action(environment, "lineage", 102)

    result = subprocess.run(
        ["bash", str(SCRIPT), "review-packet"],
        text=True,
        capture_output=True,
        env={
            **environment,
            "SLURM_JOB_ID": "103",
            "FAKE_QUALITY_SOURCE_ROUTE": "html_web",
        },
    )

    assert result.returncode != 0
    assert "document route triplet differs from the frozen logical-source roster" in result.stderr


def test_pre_review_rejects_undocumented_observed_route_from_document_parquet(
    tmp_path: Path,
) -> None:
    environment, _, _ = fixture_environment(tmp_path)
    run_action(environment, "normalize", 101)
    run_action(environment, "lineage", 102)

    result = subprocess.run(
        ["bash", str(SCRIPT), "review-packet"],
        text=True,
        capture_output=True,
        env={
            **environment,
            "SLURM_JOB_ID": "103",
            "FAKE_QUALITY_OBSERVED_ROUTE": "structured",
        },
    )

    assert result.returncode != 0
    assert "observed extraction route is not a documented secondary exception" in result.stderr


def test_pre_review_rejects_observed_route_priority_drift(
    tmp_path: Path,
) -> None:
    environment, _, _ = fixture_environment(tmp_path)
    run_action(environment, "normalize", 101)
    run_action(environment, "lineage", 102)

    result = subprocess.run(
        ["bash", str(SCRIPT), "review-packet"],
        text=True,
        capture_output=True,
        env={
            **environment,
            "SLURM_JOB_ID": "103",
            "FAKE_QUALITY_OBSERVED_PRIORITY": "secondary_exception_only",
        },
    )

    assert result.returncode != 0
    assert "observed extraction route priority does not preserve logical-source primacy" in result.stderr
