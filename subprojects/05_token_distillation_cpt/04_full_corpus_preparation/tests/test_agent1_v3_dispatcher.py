from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

PHASE = Path(__file__).resolve().parents[1]
SUBMIT = PHASE / "clariden" / "agent1_v3_submit.sh"
STAGE = PHASE / "clariden" / "agent1_v3_stage.sbatch"
GLOSSAPI_BUILDER = PHASE / "clariden" / "agent1_v3_build_glossapi_runtime.sh"
POST_ADMISSION = PHASE / "clariden" / "agent1_v3_post_admission.sh"
RUN_ID = "agent1-full-corpus-v3-20260713T123456Z-abcdef0"


def test_v3_dispatcher_actions_are_safe_dry_runs_with_cpu_resources() -> None:
    environment = os.environ.copy()
    environment["AGENT1_V3_RUN_ID"] = RUN_ID
    environment.pop("CONFIRM_LAUNCH", None)
    environment.pop("CONFIRM_CLARIDEN_CPU_EXCEPTION", None)
    expected = {
        "build-quality-runtime": "--cpus-per-task=128 --mem=240G --time=02:00:00",
        "normalize": "--cpus-per-task=128 --mem=450G --time=12:00:00",
        "lineage": "--cpus-per-task=64 --mem=450G --time=12:00:00",
        "review-packet": "--cpus-per-task=256 --mem=450G --time=12:00:00",
        "quality-review-evidence": "--cpus-per-task=64 --mem=192G --time=04:00:00",
        "admission": "--cpus-per-task=16 --mem=64G --time=02:00:00",
        "dedup": "--cpus-per-task=256 --mem=450G --time=12:00:00",
        "greekmmlu-freeze": "--cpus-per-task=16 --mem=96G --time=04:00:00",
        "decontamination": "--cpus-per-task=16 --mem=160G --time=12:00:00",
        "anonymization-sanitization": "--cpus-per-task=256 --mem=450G --time=12:00:00",
        "prestructural-freeze": "--cpus-per-task=128 --mem=450G --time=12:00:00",
    }
    for action, resources in expected.items():
        result = subprocess.run(
            ["bash", str(SUBMIT), action],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
        assert "DRY RUN:" in result.stdout
        assert f"AGENT1_V3_ACTION={action}" in result.stdout
        assert resources in result.stdout
        assert str(STAGE) in result.stdout


def test_stage_dispatcher_keeps_phase0_attempts_out_of_pre_review_actions() -> None:
    subprocess.run(["bash", "-n", str(SUBMIT)], check=True)
    subprocess.run(["bash", "-n", str(STAGE)], check=True)
    subprocess.run(["bash", "-n", str(GLOSSAPI_BUILDER)], check=True)
    subprocess.run(["bash", "-n", str(POST_ADMISSION)], check=True)
    text = STAGE.read_text(encoding="utf-8")

    # Phase 0 retains one setup per action; pre-review actions only
    # enter their own contract-first handler after dispatcher preflight.
    assert text.count("agent1_v3_phase0_attempt") == 7
    quality_build = re.search(
        r"build-quality-runtime\)\n(?P<body>.*?)\n\s*;;",
        text,
        flags=re.DOTALL,
    )
    assert quality_build is not None
    quality_checks = quality_build.group("body")
    assert "agent1_v3_phase0_attempt" in quality_checks
    assert quality_checks.index("agent1_v3_require_compute_cpu") < quality_checks.index(
        "agent1_v3_mask_gpu_visibility"
    )
    assert quality_checks.index("agent1_v3_mask_gpu_visibility") < quality_checks.index(
        "agent1_v3_require_clean_commit"
    )
    assert quality_checks.index("agent1_v3_require_clean_commit") < quality_checks.index(
        "agent1_v3_require_runtime"
    )
    assert 'agent1_v3_build_glossapi_runtime.sh"' in quality_checks

    validate_contract = re.search(
        r"validate-contract\)\n(?P<body>.*?)\n\s*;;",
        text,
        flags=re.DOTALL,
    )
    assert validate_contract is not None
    validate_checks = validate_contract.group("body")
    assert 'exec uenv run "$AGENT1_V3_UENV" --view=default --' in validate_checks
    assert '"$AGENT1_V3_RUNTIME_VENV/bin/python" "$AGENT1_V3_CONTRACT_SCRIPT" validate-run' in validate_checks
    match = re.search(
        r"normalize\|lineage\|review-packet\)\n(?P<body>.*?)\n\s*;;",
        text,
        flags=re.DOTALL,
    )
    assert match is not None
    body = match.group("body")
    assert "agent1_v3_phase0_attempt" not in body
    assert "agent1_v3_stage_preflight" in body
    assert 'agent1_v3_pre_review.sh" "$action"' in body

    stage35 = re.search(
        r"quality-review-evidence\)\n(?P<body>.*?)\n\s*;;",
        text,
        flags=re.DOTALL,
    )
    assert stage35 is not None
    stage35_body = stage35.group("body")
    assert "agent1_v3_phase0_attempt" not in stage35_body
    assert "agent1_v3_stage_preflight" in stage35_body
    assert 'agent1_v3_quality_review_evidence.sh"' in stage35_body

    post_admission = re.search(
        r"admission\|dedup\|greekmmlu-freeze\|decontamination\|anonymization-sanitization\|prestructural-freeze\)\n(?P<body>.*?)\n\s*;;",
        text,
        flags=re.DOTALL,
    )
    assert post_admission is not None
    post_admission_body = post_admission.group("body")
    assert "agent1_v3_phase0_attempt" not in post_admission_body
    assert "agent1_v3_stage_preflight" in post_admission_body
    assert 'agent1_v3_post_admission.sh" "$action"' in post_admission_body

    preflight = re.search(
        r"agent1_v3_stage_preflight\(\) \{(?P<body>.*?)\n\}",
        text,
        flags=re.DOTALL,
    )
    assert preflight is not None
    checks = preflight.group("body")
    assert checks.index("agent1_v3_require_compute_cpu") < checks.index("agent1_v3_mask_gpu_visibility")
    assert checks.index("agent1_v3_mask_gpu_visibility") < checks.index("agent1_v3_require_clean_commit")
    assert checks.index("agent1_v3_require_clean_commit") < checks.index("agent1_v3_require_runtime")


def test_glossapi_quality_builder_uses_uenv_and_no_replace_receipts() -> None:
    text = GLOSSAPI_BUILDER.read_text(encoding="utf-8")

    assert 'uenv run "$AGENT1_V3_UENV" --view=default --' in text
    assert '"$AGENT1_V3_RUNTIME_VENV/bin/python"' in text
    assert "glossapi_rs_noise glossapi_rs_cleaner" in text
    assert "--release --locked" in text
    assert "build-receipt" in text
    assert "validate-build-receipt" in text
    assert "os.O_EXCL" in text
    assert "mv -T -n --" in text
    assert "CUDA_VISIBLE_DEVICES" in text


def test_post_admission_handler_keeps_confirmation_and_postmask_stops_explicit() -> None:
    text = POST_ADMISSION.read_text(encoding="utf-8")
    assert "AGENT1_V3_ADMISSION_PROPOSAL" in text
    assert "AGENT1_V3_ADMISSION_CONFIRMATION" in text
    assert "build-packet" in text
    assert "agent1_v3_admission.py\" confirm" not in text
    assert "quarantine_and_stop" in text
    assert "verification_only" in text
    assert "second_deduplication_applied" in text
    assert "exact-reconcile" in text
    assert "partition-within-source" in text
    assert "cross-candidate" in text
    assert "candidate-to-nanochat" in text
    assert "compose-ordered-ledgers" in text
    dedup_handler = re.search(r"run_dedup\(\) \{(?P<body>.*?)\n\}\n\nrun_greekmmlu_freeze", text, flags=re.DOTALL)
    assert dedup_handler is not None
    dedup_body = dedup_handler.group("body")
    assert dedup_body.index('agent1_v3_dedup.py" exact-reconcile') < dedup_body.index(
        'agent1_v3_dedup.py" partition-within-source'
    )
    assert dedup_body.index('agent1_v3_dedup.py" partition-within-source') < dedup_body.index(
        'agent1_v3_dedup.py" filter-candidates'
    )
    assert dedup_body.index('agent1_v3_dedup.py" filter-candidates') < dedup_body.index(
        "--pass-kind candidate-to-nanochat"
    )
    assert dedup_body.index("--pass-kind candidate-to-nanochat") < dedup_body.index(
        'agent1_v3_dedup.py" compose-ordered-ledgers'
    )
    assert "identity-reconcile" not in text
    assert "prestructural-only parent run" in text
    assert "agent1_v3_transformation_waterfall.py" in text
    assert "transformation_waterfall.json" in text
