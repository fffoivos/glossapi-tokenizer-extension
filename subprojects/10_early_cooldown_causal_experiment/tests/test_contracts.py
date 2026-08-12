from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUBPROJECTS = ROOT.parent


def test_static_contract_is_self_consistent() -> None:
    subprocess.run([sys.executable, str(ROOT / "scripts/prepare_launch.py"), "--contract", str(ROOT / "configs/experiment_contract.json"), "--static-only"], check=True)


def test_single_variable_and_resource_contract() -> None:
    contract = json.loads((ROOT / "configs/experiment_contract.json").read_text())
    assert contract["single_variable_intervention"]["changed"] == "learning_rate_schedule_only"
    assert (contract["training"]["start_iteration"], contract["training"]["end_iteration"]) == (9536, 13193)
    assert contract["training"]["branch_updates"] == 3657
    assert contract["training"]["train_samples"] == 13193 * 1024
    assert contract["training"]["learning_rate"]["decay_samples"] == 3657 * 1024
    assert contract["training"]["ademamix"]["alpha_warmup_updates"] == 18284
    assert contract["training"]["ademamix"]["beta3_warmup_updates"] == 18284
    assert contract["evaluation"]["milestone_iterations"] == [10728, 11920, 13112, 13193]
    assert contract["allocation_policy"]["normal_allocations"] == 1
    assert contract["one_update_control"]["same_allocation_as_branch"] is True
    assert contract["one_update_control"]["save_checkpoint"] is False


def test_training_launcher_contains_expected_invariants() -> None:
    script = (ROOT / "clariden/train_and_gate.sbatch").read_text()
    for required in (
        "--ademamix-beta3-warmup 18284", "--ademamix-alpha-warmup 18284",
        "launch_megatron control 18722816 0 3744768", "launch_megatron branch 13509632 1 3744768",
        "MINI_SCHEDULE_ARM=D0_mixed", 'MINI_SCHEDULE_ALLOW_PREFIX="$prefix_mode"',
        '"10728,11920,13112,13193"', "--rotary-base 500000", "--rope-scaling-factor 8.0",
        "--signal=B:USR1@600", "--nodes=16", "--switches=1",
    ):
        assert required in script
    assert "--no-load-optim" not in script
    assert "--no-load-rng" not in script


def test_control_hook_is_environment_gated_and_no_save() -> None:
    hook = (SUBPROJECTS / "06_dataset_scheduling_experiments/training/exact_checkpoint_hook.py").read_text()
    assert "MINI_SCHEDULE_NO_SAVE_EXIT_ITERATION" in hook
    assert "if no_save_exit is not None and int(iteration) == no_save_exit" in hook
    ast.parse(hook)


def test_operational_jobs_obey_partition_policy() -> None:
    assert "#SBATCH --partition=normal" in (ROOT / "clariden/train_and_gate.sbatch").read_text()
    for name in ("prepare_launch_debug.sbatch", "run_checkpoint_evaluation_debug.sbatch", "run_native_endpoint_debug.sbatch", "supervise_after_training_debug.sbatch"):
        assert "#SBATCH --partition=debug" in (ROOT / "clariden" / name).read_text()


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
