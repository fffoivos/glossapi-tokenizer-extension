from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import types
from unittest.mock import patch
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_legacy_six_segment_launcher_is_executable_but_fail_closed() -> None:
    launcher = ROOT / "clariden/submit_production.sh"
    assert os.access(launcher, os.X_OK)
    result = subprocess.run([str(launcher)], text=True, capture_output=True)
    assert result.returncode == 2
    assert "retired and cannot submit jobs" in result.stderr
    assert "submit_production_resource_aware.sh" in result.stderr


def test_submitter_routes_only_training_to_normal() -> None:
    text = (ROOT / "clariden/submit_production_resource_aware.sh").read_text()
    assert "--partition=normal --time=12:00:00 --switches=1" in text
    assert "--nodes=\"$nodes\" --job-name=full8b_s0a0" in text
    assert "--partition=debug --time=00:20:00 --nodes=1" in text
    assert "supervise_campaign_resource_aware.sbatch" in text
    assert "for ((segment=" not in text
    assert "run_evaluation_queue.sbatch" not in text


def test_supervisor_uses_explicit_partitions_and_defers_next_supervisor() -> None:
    text = (ROOT / "scripts/supervise_campaign_resource_aware.py").read_text()
    assert '"--partition=normal", "--time=12:00:00"' in text
    assert '"--partition=debug", "--time=00:20:00"' in text
    assert "submit_train_only" in text
    assert "submit_evaluation_chain" in text
    assert '"next_supervisor_submission": "deferred_to_final_evaluation_job"' in text
    assert "run_evaluation_queue.sbatch" not in text
    assert 'f"FULL8_PROFILES={args.profiles}"' in text
    assert 'f"FULL8_RECIPE={args.recipe}"' in text
    assert "adopt_prequeued_train" in text
    assert "apertus_full_8b_prequeued_train_permit_v1" in text
    assert "sbatch failed after five bounded attempts" in text
    assert "time.sleep(2.0 * attempt)" in text
    assert "run_or_verify_immutable_receipt" in text
    assert "immutable receipt reproduction drift" in text
    assert "active_job_state" in text
    assert "prequeued_holder_terminal_fallback" in text


def test_terminal_prequeued_holder_falls_back_without_issuing_a_permit() -> None:
    """A timed-out holder must become a fresh leaf, never a dead permit."""

    from types import SimpleNamespace

    previous = os.environ.get("FULL8_CODE_ROOT")
    previous_contract = sys.modules.get("contract")
    contract_stub = types.ModuleType("contract")
    contract_stub.atomic_write_json = lambda *_args, **_kwargs: None
    contract_stub.read_json = lambda path: json.loads(Path(path).read_text())
    os.environ["FULL8_CODE_ROOT"] = str(ROOT.parents[1])
    sys.modules["contract"] = contract_stub
    try:
        module = load("terminal_prequeued_holder", "scripts/supervise_campaign_resource_aware.py")
    finally:
        if previous is None:
            del os.environ["FULL8_CODE_ROOT"]
        else:
            os.environ["FULL8_CODE_ROOT"] = previous
        if previous_contract is None:
            del sys.modules["contract"]
        else:
            sys.modules["contract"] = previous_contract

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        permit = root / "permit.json"
        manifest = root / "manifest.json"
        manifest.write_text(json.dumps({
            "schema_version": "apertus_full_8b_prequeued_launch_graph_v1",
            "status": "submitted",
            "segments": [{
                "segment_id": 4,
                "source_segment_id": 3,
                "source_train_job": "111",
                "start_iteration": 14627,
                "end_iteration": 18284,
                "train_job": "222",
                "permit_path": str(permit),
            }],
        }))
        args = SimpleNamespace(
            prequeued_manifest=manifest,
            run_root=root,
            segment_id=3,
            attempt=0,
        )
        recorded: list[tuple[str, dict]] = []
        with patch.object(module, "active_job_state", return_value="ABSENT"), patch.object(
            module, "slurm_state", return_value="FAILED"
        ), patch.object(
            module, "event", side_effect=lambda _args, name, value: recorded.append((name, value))
        ):
            adopted = module.adopt_prequeued_train(
                args,
                source_segment=3,
                target_segment=4,
                source_train_job="111",
                start=14627,
                end=18284,
            )

    assert adopted is None
    assert recorded == [(
        "prequeued_holder_terminal_fallback",
        {
            "prequeued_train_job": "222",
            "prequeued_live_state": "ABSENT",
            "prequeued_terminal_state": "FAILED",
            "source_segment": 3,
            "target_segment": 4,
            "updates": [14627, 18284],
        },
    )]


def test_purged_prequeued_holder_is_classified_as_absent_before_sacct() -> None:
    """Slurm uses a nonzero status for a purged, otherwise known job id."""

    previous = os.environ.get("FULL8_CODE_ROOT")
    previous_contract = sys.modules.get("contract")
    contract_stub = types.ModuleType("contract")
    contract_stub.atomic_write_json = lambda *_args, **_kwargs: None
    contract_stub.read_json = lambda path: json.loads(Path(path).read_text())
    os.environ["FULL8_CODE_ROOT"] = str(ROOT.parents[1])
    sys.modules["contract"] = contract_stub
    try:
        module = load("purged_prequeued_holder", "scripts/supervise_campaign_resource_aware.py")
    finally:
        if previous is None:
            del os.environ["FULL8_CODE_ROOT"]
        else:
            os.environ["FULL8_CODE_ROOT"] = previous
        if previous_contract is None:
            del sys.modules["contract"]
        else:
            sys.modules["contract"] = previous_contract

    result = subprocess.CompletedProcess(
        ["squeue"], 1, "", "slurm_load_jobs error: Invalid job id specified"
    )
    with patch.object(module.subprocess, "run", return_value=result):
        assert module.active_job_state("222", attempts=1) == "ABSENT"


def test_prequeued_holder_is_fail_closed_and_executes_canonical_training() -> None:
    text = (ROOT / "clariden/run_prequeued_train_holder.sbatch").read_text()
    assert "#SBATCH --partition=normal" in text
    assert "#SBATCH --time=12:00:00" in text
    assert "verify_code_bundle.py" in text
    assert "prequeued manifest binding drift" in text
    assert "launch permit binding drift" in text
    assert "FULL8_MIN_TRAIN_SECONDS" in text
    assert "FULL8_MAX_HOLD_SECONDS" in text
    assert "FULL8_ALLOCATION_RESERVE_SECONDS" in text
    assert "FULL8_MIN_TRAIN_SECONDS + FULL8_ALLOCATION_RESERVE_SECONDS" in text
    assert "holder_started_epoch=$(date +%s)" in text
    assert "schedule_successor_prequeue.py" not in text
    assert "train_segment.sbatch" in text


def test_mac_campaign_watcher_is_dynamic_read_only_and_reports_retention() -> None:
    watcher = ROOT / "clariden/watch_full8b_campaign.sh"
    text = watcher.read_text()
    assert "full8b_s1a0_hold-3037145" not in text
    assert '"RUNNING" && $4 == "normal"' in text
    assert "candidate_rows=$(squeue" in text
    assert "candidate_metric=$(grep -E 'iteration[[:space:]]+[0-9]+/'" in text
    assert 'candidate_metric_epoch=$(stat -c %Y "$candidate_log"' in text
    assert "candidate_metric_epoch > latest_metric_epoch" in text
    assert '[[ -n "$candidate_metric" ]] || continue' in text
    assert 'latest_metric="$candidate_metric"' in text
    assert "active_train=" in text
    assert 'RETENTION {"status":"unavailable"}' in text
    assert '"current_warning_candidates"' in text
    assert "sbatch" not in text
    assert "scancel" not in text


def test_prequeue_controller_is_debug_only_and_submits_one_audited_holder() -> None:
    wrapper = (ROOT / "clariden/prequeue_next_segment_debug.sbatch").read_text()
    controller = (ROOT / "scripts/prequeue_next_segment.py").read_text()
    assert "#SBATCH --partition=debug" in wrapper
    assert "prequeue controller must run on debug" in wrapper
    assert "verify_code_bundle.py" in wrapper
    assert "target segment must be the immediate successor" in controller
    assert "allowed_source_states" in controller
    assert '"--partition=normal", "--time=12:00:00"' in controller
    assert '"--switches=1"' in controller
    assert 'f"--dependency=after:{args.source_train_job}+{args.eligible_after_minutes}"' in controller
    assert "run_prequeued_train_holder.sbatch" in controller
    assert "audit_submitted_job_resources.py" in controller
    assert "fcntl.LOCK_EX" in controller
    assert "FULL8_PREQUEUE_SCHEDULE" in wrapper
    assert "sbatch failed after five bounded attempts" in controller
    assert 'parser.add_argument("--test-only", action="store_true")' in controller
    assert '"--test-only"' in controller
    assert "FULL8_ELIGIBLE_AFTER_MINUTES" in wrapper
    assert "FULL8_PREQUEUE_TEST_ONLY" in wrapper
    assert 'f"FULL8_ALLOCATION_RESERVE_SECONDS={reserve_seconds}"' in controller
    assert "train plus hold budget does not close exactly" in controller


def test_successor_holder_is_installed_by_final_evaluation_without_debug_timer() -> None:
    text = (ROOT / "evaluation/continue_checkpoint_evaluation.py").read_text()
    schedule = (ROOT / "configs/prequeue_schedule_8b.json").read_text()
    assert "prequeue_successor" in text
    assert '"--eligible-after-minutes"' in text
    assert "scripts/prequeue_next_segment.py" in text
    assert "successor_prequeue_processed" in text
    assert "prequeue_next_segment_debug.sbatch" not in text
    assert '"segment_boundaries"' in schedule
    assert "No debug timer job is used" in schedule


def test_frozen_prequeue_schedule_matches_idle_overlap_budget() -> None:
    schedule = json.loads(
        (ROOT / "configs/prequeue_schedule_8b.json").read_text()
    )
    allocation = int(schedule["allocation_seconds"])
    reserve = int(schedule["allocation_reserve_seconds"])
    targets = schedule["targets"]
    assert [int(row["target_segment_id"]) for row in targets] == [1, 2, 3, 4]

    for row in targets:
        minimum_train = int(row["minimum_train_seconds"])
        maximum_hold = int(row["maximum_hold_seconds"])
        source_minimum_train = int(row["source_minimum_train_seconds"])
        trigger = int(row["source_trigger_minutes"]) * 60

        assert minimum_train + maximum_hold + reserve == allocation
        assert 0 < trigger < source_minimum_train
        assert source_minimum_train - trigger == maximum_hold


def test_prequeued_manifest_propagates_through_evaluation_chain() -> None:
    evaluator = (ROOT / "evaluation/run_checkpoint_evaluation_debug.py").read_text()
    continuation = (ROOT / "evaluation/continue_checkpoint_evaluation.py").read_text()
    evaluator_sbatch = (ROOT / "clariden/run_checkpoint_evaluation_debug.sbatch").read_text()
    continuation_sbatch = (ROOT / "clariden/continue_checkpoint_evaluation_debug.sbatch").read_text()
    for text in (evaluator, continuation, evaluator_sbatch, continuation_sbatch):
        assert "FULL8_PREQUEUED_MANIFEST" in text
        assert "FULL8_PREQUEUE_SCHEDULE" in text


def test_debug_evaluator_is_serial_and_never_waits_for_a_child() -> None:
    text = (ROOT / "evaluation/run_checkpoint_evaluation_debug.py").read_text()
    assert '"--partition=debug"' in text
    assert "FULL8_EVAL_INDEX" in text
    assert "afterany_continuation" in text
    assert "continue_checkpoint_evaluation_debug.sbatch" in text
    assert "while True" not in text
    assert "sbatch failed after five bounded attempts" in text
    assert "run_checkpoint_native_greekmmlu.sbatch" in text
    assert "finalize_checkpoint_greekmmlu.sbatch" in text
    assert "split_per_document" in text
    assert '"--overlap"' not in text
    assert 'f"FULL8_PROFILES={args.profiles}"' in text
    assert "audit_or_cancel" in text
    assert 'subprocess.run(["scancel", job_id], check=False)' in text


def test_evaluation_continuation_recovers_and_advances_without_waiting() -> None:
    text = (ROOT / "evaluation/continue_checkpoint_evaluation.py").read_text()
    assert 'RETRYABLE = {"BOOT_FAIL", "FAILED", "NODE_FAIL"' in text
    assert "evaluation_retry_submitted" in text
    assert "next_evaluation_submitted" in text
    assert "next_segment_supervisor_submitted" in text
    assert "time.sleep(delay_seconds)" in text
    assert "while True" not in text
    assert 'f"FULL8_PROFILES={args.profiles}"' in text
    assert "write_or_verify_queue_receipt" in text
    assert "evaluation queue receipt drift" in text
    assert "audit_or_cancel" in text
    assert 'subprocess.run(["scancel", job_id], check=False)' in text
    assert "adopt_existing_supervisor_submission" in text
    assert "submitted_unverified" in text
    assert "refusing a duplicate" in text


def test_failed_child_audit_cancels_the_unaudited_job() -> None:
    module = load(
        "continue_audit_cancel",
        "evaluation/continue_checkpoint_evaluation.py",
    )
    cancelled: list[list[str]] = []

    def fail_audit(*_args, **_kwargs):
        raise ValueError("audit failed")

    def fake_run(command, **_kwargs):
        cancelled.append(command)
        return subprocess.CompletedProcess(command, 0, "", "")

    with patch.object(module, "audit_job", side_effect=fail_audit), patch.object(
        module.subprocess, "run", side_effect=fake_run
    ):
        try:
            module.audit_or_cancel(object(), role="evaluation", job_id="123")
        except ValueError:
            pass
        else:
            raise AssertionError("failed audit did not propagate")
    assert cancelled == [["scancel", "123"]]


def test_existing_supervisor_submission_is_adopted_without_resubmission() -> None:
    from types import SimpleNamespace

    module = load(
        "continue_supervisor_adoption",
        "evaluation/continue_checkpoint_evaluation.py",
    )
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        args = SimpleNamespace(
            run_root=root,
            next_segment=2,
            next_segment_start=8000,
            next_train_job="222",
            ops_root=(root / "ops").resolve(),
        )
        receipt = module.supervisor_submission_receipt(args)
        routing = root / "routing.json"
        routing.write_text("{}")
        receipt.parent.mkdir(parents=True)
        receipt.write_text(json.dumps({
            "schema_version": "apertus_full_8b_supervisor_submission_v1",
            "status": "passed",
            "next_segment": 2,
            "next_segment_start": 8000,
            "source_train_job": "222",
            "supervisor_job": "333",
            "operational_root": str(args.ops_root),
            "allocation_routing_receipt": str(routing),
        }))

        def read_json(path):
            return json.loads(Path(path).read_text())

        with patch.object(module, "slurm_state", return_value="PENDING"), patch.object(
            module, "submit", side_effect=AssertionError("duplicate submission")
        ):
            adopted = module.adopt_existing_supervisor_submission(
                args, read_json, lambda *_args, **_kwargs: None
            )
        assert adopted["supervisor_job"] == "333"


def test_ambiguous_existing_supervisor_fails_closed() -> None:
    from types import SimpleNamespace

    module = load(
        "continue_supervisor_ambiguous",
        "evaluation/continue_checkpoint_evaluation.py",
    )
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        args = SimpleNamespace(
            run_root=root,
            next_segment=2,
            next_segment_start=8000,
            next_train_job="222",
            ops_root=(root / "ops").resolve(),
        )
        receipt = module.supervisor_submission_receipt(args)
        receipt.parent.mkdir(parents=True)
        receipt.write_text(json.dumps({
            "schema_version": "apertus_full_8b_supervisor_submission_v1",
            "status": "submitted_unverified",
            "next_segment": 2,
            "next_segment_start": 8000,
            "source_train_job": "222",
            "supervisor_job": "333",
            "operational_root": str(args.ops_root),
        }))

        def read_json(path):
            return json.loads(Path(path).read_text())

        with patch.object(module, "slurm_state", return_value="UNKNOWN"):
            try:
                module.adopt_existing_supervisor_submission(
                    args, read_json, lambda *_args, **_kwargs: None
                )
            except RuntimeError as error:
                assert "refusing a duplicate" in str(error)
            else:
                raise AssertionError("ambiguous supervisor created duplicate risk")


def test_failed_supervisor_audit_cancels_and_records_rejection() -> None:
    from types import SimpleNamespace

    module = load(
        "continue_supervisor_rejection",
        "evaluation/continue_checkpoint_evaluation.py",
    )
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        args = SimpleNamespace(
            run_root=root,
            next_segment=2,
            next_segment_start=8000,
            next_train_job="222",
            ops_root=(root / "ops").resolve(),
        )
        cancelled: list[list[str]] = []

        def atomic(path, value, *, exclusive=True):
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            if exclusive and path.exists():
                raise FileExistsError(path)
            path.write_text(json.dumps(value))

        def fake_run(command, **_kwargs):
            cancelled.append(command)
            return subprocess.CompletedProcess(command, 0, "", "")

        with patch.object(module, "submit", return_value="333"), patch.object(
            module, "audit_job", side_effect=ValueError("bad routing")
        ), patch.object(module.subprocess, "run", side_effect=fake_run):
            try:
                module.submit_supervisor_with_receipt(args, atomic, ["ignored"])
            except ValueError:
                pass
            else:
                raise AssertionError("failed supervisor audit was accepted")
        value = json.loads(
            module.supervisor_submission_receipt(args).read_text()
        )
        assert value["status"] == "rejected"
        assert cancelled == [["scancel", "333"]]


def test_queue_receipt_rerun_is_idempotent_but_drift_fails() -> None:
    module = load(
        "continue_queue_receipt",
        "evaluation/continue_checkpoint_evaluation.py",
    )
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "queue.json"

        def read_json(candidate):
            return json.loads(Path(candidate).read_text())

        def write_json(candidate, value):
            Path(candidate).write_text(json.dumps(value))

        first = {
            "schema_version": "queue_v1", "status": "completed",
            "completed_at": "first", "iterations": [1, 2],
        }
        second = {**first, "completed_at": "second"}
        assert module.write_or_verify_queue_receipt(
            write_json, read_json, path, first
        ) == "written"
        assert module.write_or_verify_queue_receipt(
            write_json, read_json, path, second
        ) == "verified_existing"
        try:
            module.write_or_verify_queue_receipt(
                write_json, read_json, path, {**second, "iterations": [1, 3]}
            )
        except ValueError as error:
            assert "receipt drift" in str(error)
        else:
            raise AssertionError("queue-receipt drift was accepted")


def test_evaluation_event_updates_latest_pointer() -> None:
    from types import SimpleNamespace

    module = load(
        "continue_event_latest",
        "evaluation/continue_checkpoint_evaluation.py",
    )
    writes: list[tuple[Path, dict, bool]] = []

    def atomic(path, value, *, exclusive=True):
        writes.append((Path(path), value, exclusive))

    with tempfile.TemporaryDirectory() as directory:
        args = SimpleNamespace(run_root=Path(directory), source_segment=1)
        module.write_event(atomic, args, "test_event", {"ok": True})
    assert len(writes) == 2
    assert writes[0][0].parent.name == "events"
    assert writes[1][0].name == "latest.json"
    assert writes[1][2] is False


def prequeue_fixture(directory: str, *, existing_segments=None) -> tuple[list[str], Path]:
    root = Path(directory)
    selected = root / "selected.json"
    schedule = root / "schedule.json"
    manifest = root / "manifest.json"
    output = root / "output.json"
    selected.write_text(json.dumps({
        "selection": {
            "segment_boundaries": [0, 4000, 8000],
            "nodes": 16,
            "profile_id": "dp32_16node",
        }
    }))
    schedule.write_text(json.dumps({
        "schema_version": "apertus_full_8b_prequeue_schedule_v1",
        "status": "approved",
        "allocation_seconds": 43200,
        "allocation_reserve_seconds": 1200,
        "segment_boundaries": [0, 4000, 8000],
        "targets": [{
            "target_segment_id": 1,
            "source_minimum_train_seconds": 37800,
            "source_trigger_minutes": 560,
            "minimum_train_seconds": 37800,
            "maximum_hold_seconds": 4200,
        }],
    }))
    manifest.write_text(json.dumps({
        "schema_version": "apertus_full_8b_prequeued_launch_graph_v1",
        "status": "submitted",
        "segments": [] if existing_segments is None else existing_segments,
    }))
    argv = [
        "prequeue_next_segment.py",
        "--scientific-root", str(root / "science"),
        "--scientific-receipt", str(root / "science-receipt.json"),
        "--ops-root", str(root / "ops"),
        "--ops-receipt", str(root / "ops-receipt.json"),
        "--stage-root", str(root / "stage"),
        "--run-root", str(root / "run"),
        "--initial-megatron", str(root / "checkpoint"),
        "--selected-profile", str(selected),
        "--launch-gate", str(root / "launch.json"),
        "--prelaunch-root", str(root / "prelaunch"),
        "--recipe", str(root / "recipe.json"),
        "--profiles", str(root / "profiles.json"),
        "--train-leaf-switch", "leaf-switch-0",
        "--manifest", str(manifest),
        "--schedule", str(schedule),
        "--source-segment", "0",
        "--source-train-job", "111",
        "--target-segment", "1",
        "--minimum-train-seconds", "37800",
        "--maximum-hold-seconds", "4200",
        "--eligible-after-minutes", "560",
        "--output", str(output),
    ]
    return argv, manifest


def test_prequeue_audit_failure_cancels_job_without_manifest_row() -> None:
    module = load("prequeue_audit_failure", "scripts/prequeue_next_segment.py")
    calls: list[list[str]] = []

    def fake_run(command, **_kwargs):
        calls.append([str(value) for value in command])
        if command[0] == "squeue":
            return subprocess.CompletedProcess(command, 0, "RUNNING\n", "")
        if command[0] == "sbatch":
            return subprocess.CompletedProcess(command, 0, "12345\n", "")
        if command[0] == "scancel":
            return subprocess.CompletedProcess(command, 0, "", "")
        if str(command[0]).endswith("resolve_leaf_switch_exclusion.sh"):
            return subprocess.CompletedProcess(command, 0, "nid0001\n", "")
        if str(command[1]).endswith("audit_submitted_job_resources.py"):
            raise subprocess.CalledProcessError(1, command)
        raise AssertionError(command)

    with tempfile.TemporaryDirectory() as directory:
        argv, manifest = prequeue_fixture(directory)
        with patch.object(sys, "argv", argv), patch.object(
            module.subprocess, "run", side_effect=fake_run
        ):
            try:
                module.main()
            except subprocess.CalledProcessError:
                pass
            else:
                raise AssertionError("resource-audit failure was accepted")
        assert json.loads(manifest.read_text())["segments"] == []
        assert ["scancel", "12345"] in calls


def test_prequeue_rejects_stale_duplicate_instead_of_adopting_it() -> None:
    module = load("prequeue_stale_duplicate", "scripts/prequeue_next_segment.py")
    submitted = False

    def fake_run(command, **_kwargs):
        nonlocal submitted
        if command[0] == "squeue":
            job = command[command.index("-j") + 1]
            state = "RUNNING\n" if job == "111" else ""
            return subprocess.CompletedProcess(command, 0, state, "")
        if command[0] == "sbatch":
            submitted = True
        if str(command[0]).endswith("resolve_leaf_switch_exclusion.sh"):
            return subprocess.CompletedProcess(command, 0, "nid0001\n", "")
        raise AssertionError(command)

    existing = [{
        "segment_id": 1,
        "source_train_job": "111",
        "train_job": "999",
    }]
    with tempfile.TemporaryDirectory() as directory:
        argv, _manifest = prequeue_fixture(directory, existing_segments=existing)
        with patch.object(sys, "argv", argv), patch.object(
            module.subprocess, "run", side_effect=fake_run
        ):
            try:
                module.main()
            except RuntimeError as error:
                assert "expected PENDING or RUNNING" in str(error)
            else:
                raise AssertionError("stale prequeued job was adopted")
        assert not submitted


def test_prequeue_test_only_uses_slurm_validation_without_mutating_manifest() -> None:
    module = load("prequeue_test_only", "scripts/prequeue_next_segment.py")
    calls: list[list[str]] = []

    def fake_run(command, **_kwargs):
        calls.append([str(value) for value in command])
        if command[0] == "squeue":
            return subprocess.CompletedProcess(command, 0, "PENDING\n", "")
        if command[0] == "sbatch":
            assert "--test-only" in command
            return subprocess.CompletedProcess(command, 0, "Job 123 to start at ...\n", "")
        if str(command[0]).endswith("resolve_leaf_switch_exclusion.sh"):
            return subprocess.CompletedProcess(command, 0, "nid0001\n", "")
        raise AssertionError(command)

    with tempfile.TemporaryDirectory() as directory:
        argv, manifest = prequeue_fixture(directory)
        argv.append("--test-only")
        before = manifest.read_bytes()
        with patch.object(sys, "argv", argv), patch.object(
            module.subprocess, "run", side_effect=fake_run
        ):
            assert module.main() == 0
        assert manifest.read_bytes() == before
        assert not any(call[0] in {"scancel"} for call in calls)


def test_prequeue_rejects_delay_or_reserve_drift_before_submission() -> None:
    module = load("prequeue_policy_drift", "scripts/prequeue_next_segment.py")

    def fake_run(command, **_kwargs):
        if command[0] == "squeue":
            return subprocess.CompletedProcess(command, 0, "PENDING\n", "")
        raise AssertionError(f"unexpected external command: {command}")

    with tempfile.TemporaryDirectory() as directory:
        argv, _manifest = prequeue_fixture(directory)
        argv[argv.index("--eligible-after-minutes") + 1] = "559"
        with patch.object(sys, "argv", argv), patch.object(
            module.subprocess, "run", side_effect=fake_run
        ):
            try:
                module.main()
            except ValueError as error:
                assert "source-trigger drift" in str(error)
            else:
                raise AssertionError("wrong delayed-submission offset was accepted")

    with tempfile.TemporaryDirectory() as directory:
        argv, _manifest = prequeue_fixture(directory)
        schedule_path = Path(argv[argv.index("--schedule") + 1])
        schedule = json.loads(schedule_path.read_text())
        schedule["targets"][0]["source_minimum_train_seconds"] = 36000
        schedule_path.write_text(json.dumps(schedule))
        with patch.object(sys, "argv", argv), patch.object(
            module.subprocess, "run", side_effect=fake_run
        ):
            try:
                module.main()
            except ValueError as error:
                assert "does not preserve the target hold budget" in str(error)
            else:
                raise AssertionError(
                    "raw-step-time source budget was accepted for a delayed holder"
                )

    with tempfile.TemporaryDirectory() as directory:
        argv, _manifest = prequeue_fixture(directory)
        schedule_path = Path(argv[argv.index("--schedule") + 1])
        schedule = json.loads(schedule_path.read_text())
        schedule["allocation_reserve_seconds"] = 600
        schedule_path.write_text(json.dumps(schedule))
        with patch.object(sys, "argv", argv), patch.object(
            module.subprocess, "run", side_effect=fake_run
        ):
            try:
                module.main()
            except ValueError as error:
                assert "allocation or reserve drift" in str(error)
            else:
                raise AssertionError("reduced allocation reserve was accepted")


def test_versioned_contracts_are_required_by_all_dynamic_wrappers() -> None:
    supervisor = (ROOT / "clariden/supervise_campaign_resource_aware.sbatch").read_text()
    evaluator = (ROOT / "clariden/run_checkpoint_evaluation_debug.sbatch").read_text()
    continuation = (ROOT / "clariden/continue_checkpoint_evaluation_debug.sbatch").read_text()
    for text in (supervisor, evaluator, continuation):
        assert 'FULL8_PROFILES:?set' in text
        assert '--profiles "$FULL8_PROFILES"' in text


def test_per_document_milestones_use_two_slot_safe_inline_continuation() -> None:
    evaluator = (ROOT / "evaluation/run_checkpoint_evaluation_debug.py").read_text()
    sbatch = (ROOT / "clariden/run_checkpoint_evaluation_debug.sbatch").read_text()
    group_runner = (ROOT / "clariden/run_per_document_group_resource_aware.sh").read_text()
    finalizer = (ROOT / "evaluation/finalize_split_checkpoint_evaluation.py").read_text()
    continuation = (ROOT / "evaluation/continue_checkpoint_evaluation.py").read_text()
    auditor = (ROOT / "scripts/audit_submitted_job_resources.py").read_text()
    assert "expected_nodes = 1" in evaluator
    assert '"--array=0-3"' not in evaluator
    assert '"--time=01:25:00" if needs_per_document' in evaluator
    assert '"--gpus-per-node=4"' in evaluator
    assert "continuation_runs_per_document_inline" in evaluator
    assert "run_per_document_inline" in continuation
    assert "for group in range(4)" in continuation
    assert "run_per_document_group_resource_aware.sh" in continuation
    assert "SLURM_ARRAY_TASK_ID" in group_runner
    assert "per_document_receipts" in finalizer
    assert "len(manifest.get(\"panels\", [])) != 13" in finalizer
    assert "per_document_inline_completed" in continuation
    assert "nested_debug_submissions" in continuation
    assert "nodes * seconds <= 90 * 60" in auditor
    assert 'role == "per_document_continuation"' in auditor
    assert '"four_gpus_requested"' in auditor
    assert "#SBATCH --ntasks-per-node=1" in sbatch


def test_debug_resource_auditor_enforces_total_node_minutes() -> None:
    module = load("debug_node_minutes", "scripts/audit_submitted_job_resources.py")
    impossible = {
        "Partition": "debug", "NumNodes": "4-4", "TimeLimit": "01:15:00",
        "JobState": "PENDING",
    }
    try:
        module.audit("evaluation", "123", impossible)
    except ValueError as error:
        assert "node_minutes_fit_debug_qos" in str(error)
    else:
        raise AssertionError("four nodes for 75 minutes passed a 90 node-minute cap")

    possible = {
        "Partition": "debug", "NumNodes": "1-1", "TimeLimit": "01:25:00",
        "JobState": "PENDING",
    }
    assert module.audit("evaluation", "124", possible)["checks"][
        "node_minutes_fit_debug_qos"
    ]

    per_document = {
        "Partition": "debug", "NumNodes": "1-1", "TimeLimit": "01:25:00",
        "JobState": "PENDING", "ReqTRES": "cpu=288,gres/gpu=4,node=1",
        "NumCPUs": "288",
    }
    checks = module.audit(
        "per_document_continuation", "125", per_document
    )["checks"]
    assert checks["four_gpus_requested"]
    assert checks["full_node_cpus_requested"]


def test_split_finalizer_requires_all_13_bound_panel_receipts() -> None:
    module = load(
        "split_evaluation_finalizer",
        "evaluation/finalize_split_checkpoint_evaluation.py",
    )
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        iteration_root = root / "iter_0000010"
        attempt_root = iteration_root / "attempt_0"
        doc_root = attempt_root / "per_document"
        doc_root.mkdir(parents=True)
        panels = []
        for index in range(13):
            name = f"panel_{index:02d}"
            source = root / f"{name}.jsonl"
            source.write_text('{"doc_id":"x","text":"x"}\n')
            output = doc_root / f"{name}.documents.jsonl"
            output.write_text('{"doc_id":"x","bpb":1.0}\n')
            panels.append({
                "name": name,
                "raw_jsonl": {
                    "path": str(source), "bytes": source.stat().st_size,
                    "sha256": module.sha256_file(source),
                },
            })
            receipt = {
                "schema_version": "apertus_per_document_validation_v1",
                "status": "completed",
                "input": panels[-1]["raw_jsonl"],
                "output": {
                    "path": str(output), "bytes": output.stat().st_size,
                    "sha256": module.sha256_file(output), "rows": 1,
                },
                "aggregate": {"documents": 1},
            }
            (doc_root / f"{name}.receipt.json").write_text(json.dumps(receipt))
        manifest = root / "manifest.json"
        manifest.write_text(json.dumps({
            "schema_version": "apertus_full_8b_validation_manifest_v1",
            "status": "frozen", "panels": panels,
        }))
        greek = root / "greek.json"
        greek.write_text(json.dumps({
            "schema_version": "exact_checkpoint_native_greekmmlu_receipt_v1",
            "status": "completed", "checkpoint": {"iteration": 10},
        }))
        argv = [
            "finalize", "--iteration", "10", "--attempt", "0",
            "--iteration-root", str(iteration_root),
            "--attempt-root", str(attempt_root),
            "--greekmmlu-receipt", str(greek),
            "--validation-manifest", str(manifest),
            "--per-document-root", str(doc_root),
        ]
        with patch.object(sys, "argv", argv):
            assert module.main() == 0
        result = json.loads((iteration_root / "authoritative_attempt.json").read_text())
        assert result["status"] == "completed"
        assert result["iteration"] == 10
        assert len(result["per_document_receipts"]) == 13


def test_resource_aware_document_wrapper_preserves_slurm_gpu_ids() -> None:
    text = (ROOT / "clariden/run_per_document_group_resource_aware.sh").read_text()
    assert "assigned_devices" in text
    assert 'CUDA_VISIBLE_DEVICES=${assigned_devices[$local_rank]}' in text
    assert "score_documents_hf.py" in text
    assert "--dtype bfloat16" in text
    assert "CUDA_VISIBLE_DEVICES=$local_rank" not in text


def test_pending_supervisor_transition_is_audited_before_legacy_cancellation() -> None:
    text = (ROOT / "scripts/transition_pending_supervisor.py").read_text()
    assert 'if old_state != "PENDING"' in text
    assert '"--uenv-passthrough=ignore"' in text
    assert '"--dependency=afterany:{args.source_train_job}"' in text
    assert '"--job", f"supervisor={replacement}"' in text
    assert 'if state(replacement) != "PENDING"' in text
    assert 'if state(args.old_supervisor_job) != "PENDING"' in text
    assert 'subprocess.run(["scancel", args.old_supervisor_job], check=True)' in text
    assert 'subprocess.run(["scancel", replacement], check=False)' in text


def test_pending_supervisor_transition_accepts_only_an_audited_prior_swap() -> None:
    from types import SimpleNamespace

    module = load(
        "prior_supervisor_transition",
        "scripts/transition_pending_supervisor.py",
    )
    args = SimpleNamespace(
        segment=1,
        attempt=0,
        attempt_start=4000,
        source_train_job="3037145",
        old_supervisor_job="3037873",
    )
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        prior_ops = root / "ops-v30"
        prior_ops.mkdir()
        audit = root / "audit.json"
        audit.write_text(json.dumps({
            "schema_version": "apertus_full_8b_allocation_routing_receipt_v1",
            "status": "passed",
            "jobs": [{
                "role": "supervisor",
                "job_id": "3037873",
                "partition": "debug",
                "nodes": 1,
                "time_limit_seconds": 1200,
            }],
        }))
        prior = {
            "schema_version": "apertus_full_8b_supervisor_transition_v1",
            "status": "completed",
            "segment": 1,
            "attempt": 0,
            "attempt_start": 4000,
            "source_train_job": "3037145",
            "replacement_supervisor_job": "3037873",
            "replacement_operational_root": str(prior_ops),
            "allocation_routing_receipt": str(audit),
        }
        assert module.validate_old_supervisor_binding(args, prior) == prior_ops.resolve()

        prior["replacement_supervisor_job"] = "3037999"
        try:
            module.validate_old_supervisor_binding(args, prior)
        except ValueError as error:
            assert "prior supervisor-transition drift" in str(error)
        else:
            raise AssertionError("unbound replacement supervisor was accepted")


def test_pending_supervisor_transition_audits_replacement_before_cancel() -> None:
    module = load("pending_supervisor_transition", "scripts/transition_pending_supervisor.py")
    calls: list[list[str]] = []

    def fake_run(command, **_kwargs):
        calls.append([str(value) for value in command])
        return subprocess.CompletedProcess(command, 0, "", "")

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        old_receipt = root / "legacy.json"
        old_receipt.write_text(json.dumps({
            "schema_version": "apertus_full_8b_supervisor_submission_v1",
            "status": "passed",
            "next_segment": 1,
            "next_segment_start": 4000,
            "source_train_job": "111",
            "supervisor_job": "999",
            "operational_root": "/legacy-ops",
        }))
        output = root / "transition.json"
        argv = [
            "transition_pending_supervisor.py",
            "--scientific-root", str(root / "science"),
            "--scientific-receipt", str(root / "science-receipt.json"),
            "--ops-root", str(root / "ops"),
            "--ops-receipt", str(root / "ops-receipt.json"),
            "--stage-root", str(root / "stage"),
            "--run-root", str(root / "run"),
            "--initial-megatron", str(root / "checkpoint"),
            "--selected-profile", str(root / "selected.json"),
            "--launch-gate", str(root / "launch.json"),
            "--prelaunch-root", str(root / "prelaunch"),
            "--recipe", str(root / "recipe.json"),
            "--profiles", str(root / "profiles.json"),
            "--train-leaf-switch", "group29",
            "--prequeued-manifest", str(root / "manifest.json"),
            "--prequeue-schedule", str(root / "schedule.json"),
            "--old-supervisor-job", "999",
            "--old-supervisor-receipt", str(old_receipt),
            "--segment", "1", "--attempt", "0", "--attempt-start", "4000",
            "--source-train-job", "111", "--output", str(output),
        ]
        with patch.object(module, "verify_bundle"), patch.object(
            module, "state", return_value="PENDING"
        ), patch.object(module, "submit", return_value="222"), patch.object(
            module.subprocess, "run", side_effect=fake_run
        ), patch.object(sys, "argv", argv):
            assert module.main() == 0

        value = json.loads(output.read_text())
        assert value["status"] == "completed"
        assert value["old_supervisor_job"] == "999"
        assert value["replacement_supervisor_job"] == "222"
        audit_index = next(
            index for index, call in enumerate(calls)
            if "audit_submitted_job_resources.py" in " ".join(call)
        )
        cancel_index = calls.index(["scancel", "999"])
        assert audit_index < cancel_index


def test_missing_v19_supervisor_receipt_is_reconstructed_only_from_bound_evidence() -> None:
    module = load(
        "legacy_supervisor_receipt_bridge",
        "scripts/reconstruct_legacy_supervisor_receipt.py",
    )
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        run = root / "run"
        routing = run / "orchestration/allocation_receipts/supervisor_999.json"
        routing.parent.mkdir(parents=True)
        operational_root = (root / "legacy_ops").resolve()
        routing.write_text(json.dumps({
            "schema_version": "apertus_full_8b_allocation_routing_receipt_v1",
            "status": "passed",
            "operational_bundle": {
                "root": str(operational_root),
                "checks": {"receipt_is_frozen": True},
            },
            "jobs": [{
                "role": "supervisor", "job_id": "999", "partition": "debug",
                "nodes": 1, "time_limit_seconds": 1200, "state": "PENDING",
            }],
        }))
        latest = run / "orchestration/latest.json"
        latest.parent.mkdir(parents=True, exist_ok=True)
        latest.write_text(json.dumps({
            "event": "next_segment_supervisor_submitted",
            "source_segment": 0, "completed_iteration": 3576,
            "next_segment": 1, "next_train_job": "222", "next_supervisor_job": "999",
        }))
        queue = run / "evaluation_queues/segment_0.json"
        queue.parent.mkdir(parents=True)
        queue.write_text(json.dumps({
            "schema_version": "apertus_full_8b_evaluation_queue_v1",
            "status": "completed", "iterations": [400, 3576],
        }))
        evaluation = run / "checkpoint_evaluations/iter_0003576"
        evaluation.mkdir(parents=True)
        (evaluation / "authoritative_attempt.json").write_text(json.dumps({
            "status": "completed", "iteration": 3576,
        }))
        exact = evaluation / "attempt_0/exact_checkpoint_native_greekmmlu_receipt.json"
        exact.parent.mkdir(parents=True)
        exact.write_text(json.dumps({"status": "completed", "metrics": {"n": 7}}))
        manifest = run / "submissions/prequeued_launch_graph_v1.json"
        manifest.parent.mkdir(parents=True)
        manifest.write_text(json.dumps({"segments": [{
            "segment_id": 1, "train_job": "222", "start_iteration": 4000,
        }]}))
        output = run / "orchestration/supervisor_submission_receipts/segment_1.json"
        argv = [
            "reconstruct_legacy_supervisor_receipt.py", "--run-root", str(run),
            "--prequeued-manifest", str(manifest),
            "--allocation-routing-receipt", str(routing),
            "--operational-root", str(operational_root), "--supervisor-job", "999",
            "--next-train-job", "222", "--source-segment", "0", "--next-segment", "1",
            "--next-segment-start", "4000", "--completed-iteration", "3576",
            "--output", str(output),
        ]
        scontrol = (
            "JobId=999 JobState=PENDING Dependency=afterany:222(unfulfilled) "
            "TimeLimit=00:20:00 Partition=debug NumNodes=1-1 "
            f"Command={operational_root}/clariden/supervise_campaign_resource_aware.sbatch"
        )
        with patch.object(
            module.subprocess, "run",
            return_value=subprocess.CompletedProcess(["scontrol"], 0, scontrol, ""),
        ), patch.object(sys, "argv", argv):
            assert module.main() == 0
        receipt = json.loads(output.read_text())
        assert receipt["status"] == "passed"
        assert receipt["supervisor_job"] == "999"
        assert receipt["source_train_job"] == "222"
        assert receipt["reconstruction"] == "v19_missing_supervisor_receipt_bridge_v1"


def test_mac_transition_watcher_only_calls_the_receipt_bound_helper() -> None:
    text = (ROOT / "clariden/watch_pending_supervisor_transition_v29.sh").read_text()
    assert "FULL8_WATCH_MAX_CHECKS" in text
    # The watcher is launched by macOS /bin/bash 3.2, which has no readarray.
    assert "readarray" not in text
    assert "transition_pending_supervisor.py" in text
    assert "reconstruct_legacy_supervisor_receipt.py" in text
    assert 'debug_rows=$(squeue -h -u fffoivos -p debug' in text
    assert '[[ "$debug_rows" != "$old_job|PENDING" ]]' in text
    assert '"$result" == DONE* || "$result" == TRANSITIONED*' in text
    assert "WAIT ssh_or_remote_error" in text
    assert "date --iso-8601" not in text
    assert "sbatch " not in text


def test_campaign_watcher_is_read_only_and_terminal_receipt_bound() -> None:
    text = (ROOT / "clariden/watch_full8b_campaign.sh").read_text()
    assert "FULL8_CAMPAIGN_WATCH_MAX_CHECKS" in text
    assert "training_completion_receipt.json" in text
    assert "full8b_s" in text
    assert "latest_metric" in text
    assert "latest_stderr" in text
    assert "iteration[[:space:]]+[0-9]+" in text
    assert "sbatch " not in text
    assert "scancel " not in text
    assert "rsync " not in text
    assert "ssh -o BatchMode=yes" in text


def test_all_auxiliary_sbatch_headers_are_debug() -> None:
    for relative in (
        "clariden/supervise_campaign_resource_aware.sbatch",
        "clariden/run_checkpoint_evaluation_debug.sbatch",
        "clariden/continue_checkpoint_evaluation_debug.sbatch",
        "clariden/finalize_and_submit_production_resource_aware.sbatch",
        "clariden/prove_resource_aware_routing.sbatch",
        "clariden/resource_aware_routing_child.sbatch",
        "clariden/prove_evaluation_overlap.sbatch",
        "clariden/prove_successor_launch_gate_debug.sbatch",
        "clariden/prepare_successor_stage_debug.sbatch",
        "clariden/prepare_successor_contracts_debug.sbatch",
    ):
        text = (ROOT / relative).read_text()
        assert "#SBATCH --partition=debug" in text
        assert "SLURM_JOB_PARTITION" in text


def test_successor_launch_gate_delegates_to_the_original_gate() -> None:
    text = (ROOT / "scripts/build_successor_launch_gate.py").read_text()
    assert "import build_launch_gate as v45_gate" in text
    assert "base_gate_completed_all_original_checks" in text
    assert '"data.sanitized_source_receipt"' in text
    assert '"data.eligibility_policy.proof"' in text
    assert "v45_gate.main()" in text
    assert '"--code-root", str(code_root)' in text
    assert '"--recipe", str(args.recipe)' in text
    assert '"--profiles", str(args.profiles)' in text
    assert "restart_control_from_promotion" in text
    assert "DP32 parity receipt" in text
    assert "v45_restart_control_schema_bug_corrected_from_bound_parity_receipt" in text


def test_successor_rebind_has_its_required_copy_import() -> None:
    text = (ROOT / "scripts/rebind_selected_execution_profile.py").read_text()
    assert "import copy" in text
    assert "copy.deepcopy(old)" in text


def test_finalizer_uses_the_successor_adapter_not_a_weakened_train_launch() -> None:
    text = (ROOT / "clariden/finalize_and_submit_production_resource_aware.sbatch").read_text()
    assert "build_successor_launch_gate.py" in text
    assert "--source-selected-profile" in text
    assert "--source-promotion-receipt" in text
    assert "--stage-identity" in text


def test_successor_gate_proof_is_debug_only_and_uses_the_same_adapter() -> None:
    text = (ROOT / "clariden/prove_successor_launch_gate_debug.sbatch").read_text()
    assert "#SBATCH --partition=debug" in text
    assert "build_successor_launch_gate.py" in text
    assert "capture_launch_environment.py" in text
    assert "submit_production_resource_aware.sh" not in text


def test_resource_auditor_accepts_only_frozen_geometries() -> None:
    module = load("resource_audit", "scripts/audit_submitted_job_resources.py")
    train = module.audit("train", "1", {
        "Partition": "normal", "NumNodes": "16", "TimeLimit": "12:00:00",
        "ReqSwitch": "1@3600", "JobState": "PENDING",
    })
    debug = module.audit("supervisor", "2", {
        "Partition": "debug", "NumNodes": "1", "TimeLimit": "00:20:00",
        "JobState": "PENDING",
    })
    assert train["partition"] == "normal"
    assert debug["partition"] == "debug"


def test_resource_auditor_accepts_only_exact_pending_node_ranges() -> None:
    module = load("resource_audit_pending_range", "scripts/audit_submitted_job_resources.py")
    assert module.parse_exact_nodes("16-16") == 16
    try:
        module.parse_exact_nodes("1-16")
    except ValueError:
        pass
    else:
        raise AssertionError("a non-exact Slurm node range was accepted")


def test_resource_auditor_accepts_only_one_switch_in_clariden_scontrol_format() -> None:
    module = load("resource_audit_switches", "scripts/audit_submitted_job_resources.py")
    accepted = module.audit("train", "1", {
        "Partition": "normal", "NumNodes": "16-16", "TimeLimit": "12:00:00",
        "Switches": "1@00:05:00", "JobState": "PENDING",
    })
    assert accepted["switches"] == "1@00:05:00"
    try:
        module.audit("train", "2", {
            "Partition": "normal", "NumNodes": "16", "TimeLimit": "12:00:00",
            "Switches": "2@00:05:00", "JobState": "PENDING",
        })
    except ValueError:
        pass
    else:
        raise AssertionError("a multi-switch request was accepted")


def test_resource_auditor_rejects_auxiliary_normal_job() -> None:
    module = load("resource_audit_reject", "scripts/audit_submitted_job_resources.py")
    try:
        module.audit("evaluation", "3", {
            "Partition": "normal", "NumNodes": "1", "TimeLimit": "01:00:00",
            "JobState": "PENDING",
        })
    except ValueError:
        pass
    else:
        raise AssertionError("normal auxiliary job was accepted")
