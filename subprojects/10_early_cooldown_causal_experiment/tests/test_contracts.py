from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUBPROJECTS = ROOT.parent


def test_static_contract_is_self_consistent() -> None:
    subprocess.run([sys.executable, str(ROOT / "scripts/prepare_launch.py"), "--contract", str(ROOT / "configs/experiment_contract.json"), "--static-only"], check=True)


def test_single_variable_and_resource_contract() -> None:
    contract = json.loads((ROOT / "configs/experiment_contract.json").read_text())
    assert contract["single_variable_intervention"]["changed"] == "learning_rate_schedule_only"
    assert (contract["training"]["start_iteration"], contract["training"]["intervention_iteration"], contract["training"]["end_iteration"]) == (9536, 9536, 13193)
    assert contract["training"]["branch_updates"] == 3657
    assert contract["training"]["train_samples"] == 13193 * 1024
    assert contract["training"]["learning_rate"]["decay_samples"] == 3657 * 1024
    assert contract["training"]["ademamix"]["alpha_warmup_updates"] == 18284
    assert contract["training"]["ademamix"]["beta3_warmup_updates"] == 18284
    assert contract["evaluation"]["milestone_iterations"] == [10728, 11920, 13112, 13193]
    assert contract["allocation_policy"]["normal_allocations"] == 1
    sandwich = contract["sandwich_same_allocation_gate"]
    assert sandwich["same_nodes_and_allocation"] is True
    assert sandwich["control_order"] == ["parent_peak_before", "intervention", "parent_peak_after"]
    assert sandwich["maximum_parent_control_display_spread"] == 0.001
    assert sandwich["historical_absolute_gradient_is_not_an_acceptance_criterion"] is True


def test_training_launcher_contains_expected_invariants() -> None:
    script = (ROOT / "clariden/train_and_gate.sbatch").read_text()
    for required in (
        "--ademamix-beta3-warmup 18284", "--ademamix-alpha-warmup 18284",
        "launch_megatron reference_peak_before_probe 18722816 1 3744768",
        "launch_megatron intervention_cooldown_probe 13509632 1 3744768",
        "launch_megatron reference_peak_after_probe 18722816 1 3744768",
        "launch_megatron branch 13509632 1 3744768",
        "MINI_SCHEDULE_ARM=D0_mixed", 'MINI_SCHEDULE_ALLOW_PREFIX="$prefix_mode"',
        '"10728,11920,13112,13193"', "--rotary-base 500000", "--rope-scaling-factor 8.0",
        "--signal=B:USR1@600", "--nodes=16", "--switches=1",
    ):
        assert required in script
    assert "--no-load-optim" not in script
    assert "--no-load-rng" not in script


def test_single_allocation_budget_closes() -> None:
    contract = json.loads((ROOT / "configs/experiment_contract.json").read_text())
    policy = contract["allocation_policy"]
    assert policy["branch_conservative_runtime_seconds"] + policy["branch_reserve_seconds"] == 12 * 3600
    submit = (ROOT / "clariden/submit_experiment.sh").read_text()
    assert "--role training" in submit
    assert "--time=12:00:00" in submit
    assert '"schema_version": "apertus_full8_early_cooldown_launch_graph_v3"' in (ROOT / "scripts/freeze_launch_graph.py").read_text()
    assert "after:$replay" not in submit


def test_scheduler_can_choose_any_eligible_single_leaf() -> None:
    submit = (ROOT / "clariden/submit_experiment.sh").read_text()
    prepare = (ROOT / "clariden/prepare_launch_debug.sbatch").read_text()
    supervisor = (ROOT / "clariden/supervise_after_training_debug.sbatch").read_text()
    training = (ROOT / "clariden/train_and_gate.sbatch").read_text()
    snapshot = (ROOT / "scripts/capture_scheduler_snapshot.py").read_text()
    for script in (submit, prepare, supervisor):
        assert "--switches=1" in script
        assert "--exclude=" not in script
        assert "EARLY_TRAIN_LEAF_SWITCH" not in script
        assert "resolve_leaf_switch_exclusion.sh" not in script
    assert "EARLY_TRAIN_LEAF_SWITCH" not in training
    assert '[[ ${#leaves[@]} == 1 ]]' in training
    assert '"selection": "scheduler_selected"' in snapshot
    assert "group29" not in snapshot


def test_control_hook_is_environment_gated_and_no_save() -> None:
    hook = (SUBPROJECTS / "06_dataset_scheduling_experiments/training/exact_checkpoint_hook.py").read_text()
    assert "MINI_SCHEDULE_NO_SAVE_EXIT_ITERATION" in hook
    assert "if no_save_exit is not None and int(iteration) == no_save_exit" in hook
    ast.parse(hook)


def test_operational_jobs_obey_partition_policy() -> None:
    assert "#SBATCH --partition=normal" in (ROOT / "clariden/train_and_gate.sbatch").read_text()
    for name in ("freeze_bundle_debug.sbatch", "prepare_launch_debug.sbatch", "run_checkpoint_evaluation_debug.sbatch", "run_native_endpoint_debug.sbatch", "supervise_after_training_debug.sbatch"):
        assert "#SBATCH --partition=debug" in (ROOT / "clariden" / name).read_text()
    deploy = (ROOT / "clariden/deploy_bundle.sh").read_text()
    assert "cp -a \"$BASE\" \"$REMOTE_ROOT\"" not in deploy
    assert "freeze_code_bundle.py" not in deploy
    assert "freeze_bundle_debug.sbatch" in deploy


def test_readonly_watcher_never_mutates_slurm_or_executes_data_work() -> None:
    watcher = (ROOT / "clariden/watch_early_cooldown_readonly.sh").read_text()
    assert "squeue" in watcher
    assert "sbatch" not in watcher
    assert "scancel" not in watcher
    for forbidden in ("dedup", "anonymize", "decontaminate", "tokenize", "pack_catalog", "rsync"):
        assert forbidden not in watcher
    assert "find \"$run_root/checkpoint_evaluations\"" in watcher
    assert "|| true; } | wc -l" in watcher


def test_macos_watcher_is_readonly_and_plist_is_bounded() -> None:
    watcher = (ROOT / "clariden/watch_early_cooldown_macos.sh").read_text()
    plist = (ROOT / "clariden/com.fffoivos.apertus-early-cooldown-watch.plist").read_text()
    assert "/usr/bin/ssh" in watcher
    assert "squeue" in watcher and "sacct" in watcher
    assert "sbatch" not in watcher and "scancel" not in watcher
    assert "StartInterval" in plist and "<integer>300</integer>" in plist
    assert "RunAtLoad" in plist


def test_supervisor_handles_training_completion_and_bounded_recovery() -> None:
    supervisor = (ROOT / "clariden/supervise_after_training_debug.sbatch").read_text()
    assert "branch_completed" in supervisor
    assert "branch_gracefully_stopped" in supervisor
    assert "EARLY_RECOVERY_MODE=1" in supervisor
    assert "branch_holder_expired" not in supervisor


def test_no_data_processing_or_copy_is_in_launch_path() -> None:
    submit = (ROOT / "clariden/submit_experiment.sh").read_text().lower()
    for forbidden in ("dedup", "anonymize", "decontaminate", "tokenize", "pack_catalog", "build_packing", "rsync"):
        assert forbidden not in submit


def test_python_and_shell_sources_parse() -> None:
    for path in ROOT.rglob("*.py"):
        ast.parse(path.read_text(), filename=str(path))
    for path in list((ROOT / "clariden").glob("*.sh")) + list((ROOT / "clariden").glob("*.sbatch")):
        subprocess.run(["bash", "-n", str(path)], check=True)


def test_slurm_singleton_node_range_is_accepted_but_variable_range_is_not() -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    from audit_submitted_job import exact_node_count

    assert exact_node_count("16") == 16
    assert exact_node_count("16-16") == 16
    try:
        exact_node_count("1-16")
    except ValueError:
        pass
    else:
        raise AssertionError("variable node range should fail closed")


def test_training_job_audit_checks_single_leaf_request_without_node_pinning() -> None:
    audit = (ROOT / "scripts/audit_submitted_job.py").read_text()
    assert 'fields.get("Switches", "").startswith("1@")' in audit
    assert 'fields.get("ExcNodeList") in {None, "(null)"}' in audit
    assert '"switches": fields.get("Switches")' in audit
    assert '"excluded_nodes": fields.get("ExcNodeList")' in audit


def _metric_line(iteration: int, row: dict[str, float]) -> str:
    return f"iteration {iteration:8d}/   18284 | " + " | ".join(f"{key}: {value}" for key, value in row.items()) + " |\n"


def test_sandwich_gate_uses_concurrent_control_envelope_and_exact_other_fields() -> None:
    contract = json.loads((ROOT / "configs/experiment_contract.json").read_text())
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        row = {
            "learning rate": 5.5e-5,
            "consumed samples": 9765888,
            "consumed tokens": 40.001,
            "global batch size": 1024,
            "lm loss": 1.7,
            "base-token target loss": 1.48,
            "base-token target count": 186214.6,
            "base-token target bytes": 826796.2,
            "added-token target loss": 2.27,
            "added-token target count": 71786.56,
            "added-token target bytes": 830928.5,
            "grad norm": 0.669,
            "params norm": 6841.339,
            "loss scale": 1.0,
            "number of skipped iterations": 0,
            "number of nan iterations": 0,
        }
        reference_before_log = tmp / "reference_before.log"
        reference_before_log.write_text(_metric_line(9537, row))
        intervention_row = dict(row)
        intervention_row["learning rate"] = 5.4e-5
        intervention_row["grad norm"] = 0.668
        intervention_log = tmp / "intervention.log"
        intervention_log.write_text(_metric_line(9537, intervention_row))
        reference_after_row = dict(row)
        reference_after_row["grad norm"] = 0.668
        reference_after_log = tmp / "reference_after.log"
        reference_after_log.write_text(_metric_line(9537, reference_after_row))
        checkpoint_root = tmp / "checkpoints"
        metadata = checkpoint_root / "iter_0009537/.metadata"
        metadata.parent.mkdir(parents=True)
        metadata.write_text("complete")
        (metadata.parent / "common.pt").write_text("complete")
        code_receipt = tmp / "bundle.json"
        code_receipt.write_text("{}")
        source_receipt = tmp / "source.json"
        source_files = [{"relative_path": f"__{index}.distcp"} for index in range(129)]
        source_files.extend([{"relative_path": ".metadata"}, {"relative_path": "common.pt"}])
        source_receipt.write_text(json.dumps({"schema_version": "megatron_exact_checkpoint_view_v1", "iteration": 9536, "source_files": source_files}))
        contract["parent"]["checkpoint_receipt"] = {
            "path": str(source_receipt),
            "sha256": hashlib.sha256(source_receipt.read_bytes()).hexdigest(),
        }
        contract_path = tmp / "contract.json"
        contract_path.write_text(json.dumps(contract))
        branch_receipt = tmp / "branch_receipt.json"
        subprocess.run([
            sys.executable, str(ROOT / "scripts/finalize_branch_restart_control.py"),
            "--contract", str(contract_path),
            "--reference-before-log", str(reference_before_log),
            "--intervention-log", str(intervention_log),
            "--reference-after-log", str(reference_after_log),
            "--intervention-checkpoint-root", str(checkpoint_root),
            "--source-checkpoint-receipt", str(source_receipt),
            "--code-bundle-receipt", str(code_receipt), "--output", str(branch_receipt),
        ], check=True)
        assert json.loads(branch_receipt.read_text())["status"] == "passed"
        intervention_row["grad norm"] = 0.667
        intervention_log.write_text(_metric_line(9537, intervention_row))
        failed = subprocess.run([
            sys.executable, str(ROOT / "scripts/finalize_branch_restart_control.py"),
            "--contract", str(contract_path),
            "--reference-before-log", str(reference_before_log),
            "--intervention-log", str(intervention_log),
            "--reference-after-log", str(reference_after_log),
            "--intervention-checkpoint-root", str(checkpoint_root),
            "--source-checkpoint-receipt", str(source_receipt),
            "--code-bundle-receipt", str(code_receipt), "--output", str(tmp / "failed.json"),
        ], capture_output=True, text=True)
        assert failed.returncode != 0
        intervention_row["grad norm"] = 0.668
        intervention_log.write_text(_metric_line(9537, intervention_row))
        reference_after_row["grad norm"] = 0.667
        reference_after_log.write_text(_metric_line(9537, reference_after_row))
        failed_wide_control = subprocess.run([
            sys.executable, str(ROOT / "scripts/finalize_branch_restart_control.py"),
            "--contract", str(contract_path),
            "--reference-before-log", str(reference_before_log),
            "--intervention-log", str(intervention_log),
            "--reference-after-log", str(reference_after_log),
            "--intervention-checkpoint-root", str(checkpoint_root),
            "--source-checkpoint-receipt", str(source_receipt),
            "--code-bundle-receipt", str(code_receipt), "--output", str(tmp / "failed_wide_control.json"),
        ], capture_output=True, text=True)
        assert failed_wide_control.returncode != 0
        reference_after_row["grad norm"] = 0.668
        reference_after_log.write_text(_metric_line(9537, reference_after_row))
        intervention_row["consumed samples"] = 9766912
        intervention_log.write_text(_metric_line(9537, intervention_row))
        failed_exact = subprocess.run([
            sys.executable, str(ROOT / "scripts/finalize_branch_restart_control.py"),
            "--contract", str(contract_path),
            "--reference-before-log", str(reference_before_log),
            "--intervention-log", str(intervention_log),
            "--reference-after-log", str(reference_after_log),
            "--intervention-checkpoint-root", str(checkpoint_root),
            "--source-checkpoint-receipt", str(source_receipt),
            "--code-bundle-receipt", str(code_receipt), "--output", str(tmp / "failed_exact.json"),
        ], capture_output=True, text=True)
        assert failed_exact.returncode != 0


def test_sandwich_gate_contract_bounds_control_spread_to_logger_quantum() -> None:
    gate = json.loads((ROOT / "configs/experiment_contract.json").read_text())[
        "sandwich_same_allocation_gate"
    ]
    assert gate["gradient_display_precision_decimals"] == 3
    assert gate["maximum_parent_control_display_spread"] == 10 ** (
        -gate["gradient_display_precision_decimals"]
    )
    assert gate["intervention_gradient_must_be_inside_parent_control_envelope"] is True
