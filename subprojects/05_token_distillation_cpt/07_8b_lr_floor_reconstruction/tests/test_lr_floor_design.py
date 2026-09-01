from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECIPE = json.loads((ROOT / "configs" / "recipe_13b_lr_floor.json").read_text())


def load_validator():
    spec = importlib.util.spec_from_file_location("lr13_validator", ROOT / "scripts_validate_recipe.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_lr_floor_wrapper():
    path = ROOT / "train" / "runtime_patches" / "lr_floor_resume.py"
    spec = importlib.util.spec_from_file_location("lr_floor_resume", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_recipe_contract():
    load_validator().validate(RECIPE)


def test_only_lr_floor_changes_between_tails():
    training = RECIPE["training"]
    assert training["lr_peak"] == "5.5e-5"
    assert training["lr_warmup_init"] == "5.5e-6"
    assert training["lr_warmup_iterations"] == 400
    assert training["lr_decay"] == "1-sqrt"
    assert training["lr_decay_samples"] == 659179
    assert set(training["lr_floors"]) == {"T10", "T20", "T30"}


def test_shared_prefix_precedes_first_divergent_update():
    geometry = RECIPE["geometry"]
    branch = geometry["cooldown_branch_iteration"]
    threshold = geometry["cooldown_threshold_samples"]
    gbs = geometry["global_batch_sequences"]
    assert branch * gbs < threshold <= (branch + 1) * gbs


def test_tail_average_checkpoints_are_equal_token_intervals():
    geometry = RECIPE["geometry"]
    checkpoints = geometry["tail_averaging_checkpoints"]
    gaps = [b - a for a, b in zip(checkpoints, checkpoints[1:])]
    assert gaps == [107] * 5
    assert checkpoints[-1] < geometry["terminal_iteration"]


def test_phase_data_are_same_for_all_lr_tails():
    assert RECIPE["phases"]["phase_2"]["iteration_end"] == 3218
    assert RECIPE["geometry"]["cooldown_branch_iteration"] == 2574
    assert RECIPE["phases"]["phase_2"]["mix_exact"] == {
        "non_hplt_new_greek": "0.740740740740740741",
        "foreign_replay": "0.222222222222222222",
        "old_greek_replay": "0.037037037037037037",
    }


def test_xfer_freezer_is_staged_on_capstor_and_python36_compatible():
    launcher = (ROOT / "train" / "submit_three_lr_tails.sh").read_text()
    freezer = (ROOT / "train" / "freeze_resume_checkpoint.py").read_text()
    assert 'freeze_bundle="$submissions/freeze_bundle"' in launcher
    assert launcher.count('"$freeze_bundle/clariden/freeze_checkpoint.sbatch"') == 4
    assert "from __future__ import annotations" not in freezer


def test_resumed_optimizer_param_groups_receive_branch_floor():
    module = load_lr_floor_wrapper()

    class Scheduler:
        min_lr = 5.5e-6

    scheduler = Scheduler()
    param_group = {"min_lr": 5.5e-6}
    assert module.enforce_lr_floor(scheduler, param_group, 1.65e-5)
    assert scheduler.min_lr == 1.65e-5
    assert param_group["min_lr"] == 1.65e-5
    assert not module.enforce_lr_floor(scheduler, param_group, 1.65e-5)


def test_lr_floor_wrapper_is_exported_by_training_config():
    config = (ROOT / "train" / "lr_floor_config.env").read_text()
    assert "train/runtime_patches/lr_floor_resume.py" in config
    assert 'CPT_MIN_LR_OVERRIDE="$LR_FINAL"' in config
    assert "export CPT_PHASE_START_ITERATION CPT_GLOBAL_BATCH_SIZE CPT_MIN_LR_OVERRIDE" in config


def test_lr_floor_wrapper_is_part_of_future_frozen_assets():
    freezer = (ROOT / "dataset" / "freeze_training_assets.py").read_text()
    assert 'code_root / "train" / "runtime_patches" / "lr_floor_resume.py"' in freezer


def test_current_run_recovery_keeps_original_assets_preflight():
    sbatch = (ROOT / "clariden" / "train_segment_lr_floor_resume_recovery.sbatch").read_text()
    adapter = (ROOT / "train" / "runtime_patches" / "lr_floor_config_recovery.env").read_text()
    assert 'preflight_segment.py' in sbatch
    assert '--assets "$LR13_ASSETS_RECEIPT"' in sbatch
    assert 'TRAIN_CONFIG_OVERRIDE="$LR13_RECOVERY_BUNDLE/lr_floor_config_recovery.env"' in sbatch
    assert 'source "$LR13_CODE_ROOT/train/lr_floor_config.env"' in adapter
    assert 'TRAINER_WRAPPER="$LR13_RECOVERY_BUNDLE/lr_floor_resume.py"' in adapter
