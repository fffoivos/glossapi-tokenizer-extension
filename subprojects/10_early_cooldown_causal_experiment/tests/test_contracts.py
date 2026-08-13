from __future__ import annotations

import ast
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
    assert (contract["training"]["start_iteration"], contract["training"]["intervention_iteration"], contract["training"]["end_iteration"]) == (8000, 9536, 13193)
    assert contract["training"]["replay_updates"] == 1536
    assert contract["training"]["branch_updates"] == 3657
    assert contract["training"]["train_samples"] == 13193 * 1024
    assert contract["training"]["learning_rate"]["decay_samples"] == 3657 * 1024
    assert contract["training"]["ademamix"]["alpha_warmup_updates"] == 18284
    assert contract["training"]["ademamix"]["beta3_warmup_updates"] == 18284
    assert contract["evaluation"]["milestone_iterations"] == [10728, 11920, 13112, 13193]
    assert contract["allocation_policy"]["normal_allocations"] == 2
    assert contract["one_update_control"]["same_allocation_as_replay"] is True
    assert contract["one_update_control"]["save_checkpoint"] is False


def test_training_launcher_contains_expected_invariants() -> None:
    script = (ROOT / "clariden/train_and_gate.sbatch").read_text()
    for required in (
        "--ademamix-beta3-warmup 18284", "--ademamix-alpha-warmup 18284",
        "launch_megatron replay_control 18722816 0 3744768", "launch_megatron replay 18722816 0 3744768",
        "launch_megatron branch_control 13509632 1 3744768", "launch_megatron branch 13509632 1 3744768",
        "MINI_SCHEDULE_ARM=D0_mixed", 'MINI_SCHEDULE_ALLOW_PREFIX="$prefix_mode"',
        '"10728,11920,13112,13193"', "--rotary-base 500000", "--rope-scaling-factor 8.0",
        "--signal=B:USR1@600", "--nodes=16", "--switches=1",
    ):
        assert required in script
    assert "--no-load-optim" not in script
    assert "--no-load-rng" not in script


def test_replay_and_branch_holder_budgets_close() -> None:
    contract = json.loads((ROOT / "configs/experiment_contract.json").read_text())
    policy = contract["allocation_policy"]
    assert policy["branch_minimum_runtime_seconds"] + policy["branch_maximum_hold_seconds"] + policy["branch_reserve_seconds"] == 12 * 3600
    assert policy["replay_conservative_runtime_seconds"] - policy["branch_source_trigger_seconds"] == policy["branch_maximum_hold_seconds"]
    submit = (ROOT / "clariden/submit_experiment.sh").read_text()
    assert '--dependency="after:$replay+200"' in submit
    assert "--role branch_holder" in submit
    assert '"schema_version": "apertus_full8_early_cooldown_launch_graph_v2"' in (ROOT / "scripts/freeze_launch_graph.py").read_text()


def test_control_hook_is_environment_gated_and_no_save() -> None:
    hook = (SUBPROJECTS / "06_dataset_scheduling_experiments/training/exact_checkpoint_hook.py").read_text()
    assert "MINI_SCHEDULE_NO_SAVE_EXIT_ITERATION" in hook
    assert "if no_save_exit is not None and int(iteration) == no_save_exit" in hook
    ast.parse(hook)


def test_operational_jobs_obey_partition_policy() -> None:
    assert "#SBATCH --partition=normal" in (ROOT / "clariden/train_and_gate.sbatch").read_text()
    for name in ("prepare_launch_debug.sbatch", "run_checkpoint_evaluation_debug.sbatch", "run_native_endpoint_debug.sbatch", "supervise_after_training_debug.sbatch"):
        assert "#SBATCH --partition=debug" in (ROOT / "clariden" / name).read_text()


def test_supervisor_handles_clean_holder_expiry_without_racing_fallback() -> None:
    supervisor = (ROOT / "clariden/supervise_after_training_debug.sbatch").read_text()
    assert "delegated_to_branch_supervisor_fallback" in supervisor
    assert "branch_holder_expired" in supervisor


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


def _metric_line(iteration: int, row: dict[str, float]) -> str:
    return f"iteration {iteration:8d}/   18284 | " + " | ".join(f"{key}: {value}" for key, value in row.items()) + " |\n"


def test_parent_replay_and_branch_restart_finalizers_pass_synthetic_exact_rows() -> None:
    contract = json.loads((ROOT / "configs/experiment_contract.json").read_text())
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        replay_log = tmp / "replay.log"
        replay_log.write_text("".join(_metric_line(int(key), row) for key, row in contract["parent_replay_gate"]["expected"].items()))
        checkpoint_root = tmp / "checkpoints"
        metadata = checkpoint_root / "iter_0009536/.metadata"
        metadata.parent.mkdir(parents=True)
        metadata.write_text("complete")
        code_receipt = tmp / "bundle.json"
        code_receipt.write_text("{}")
        replay_receipt = tmp / "replay_receipt.json"
        subprocess.run([
            sys.executable, str(ROOT / "scripts/finalize_parent_replay.py"),
            "--contract", str(ROOT / "configs/experiment_contract.json"),
            "--replay-log", str(replay_log), "--checkpoint-root", str(checkpoint_root),
            "--code-bundle-receipt", str(code_receipt), "--output", str(replay_receipt),
        ], check=True)
        branch_row = dict(contract["parent_replay_gate"]["expected"]["9537"])
        branch_row["learning rate"] = 5.4e-5
        branch_log = tmp / "branch.log"
        branch_log.write_text(_metric_line(9537, branch_row))
        branch_receipt = tmp / "branch_receipt.json"
        subprocess.run([
            sys.executable, str(ROOT / "scripts/finalize_branch_restart_control.py"),
            "--contract", str(ROOT / "configs/experiment_contract.json"),
            "--control-log", str(branch_log), "--parent-replay-receipt", str(replay_receipt),
            "--code-bundle-receipt", str(code_receipt), "--output", str(branch_receipt),
        ], check=True)
        assert json.loads(branch_receipt.read_text())["status"] == "passed"
