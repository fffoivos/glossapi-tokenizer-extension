from __future__ import annotations

import stat
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from freeze_phase_blend_cache import (
    compare_cache_inventory,
    tree_inventory,
)
from materialize_phase_cache import clone_declared_files


def test_cache_inventory_classifies_added_changed_and_missing(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    root.mkdir()
    (root / "kept.npy").write_bytes(b"kept")
    (root / "changed.npy").write_bytes(b"original")
    (root / "missing.npy").write_bytes(b"missing")
    expected, _ = tree_inventory(root)

    (root / "changed.npy").write_bytes(b"different")
    (root / "missing.npy").unlink()
    (root / "added.npy").write_bytes(b"added")
    drift = compare_cache_inventory(root, expected)

    assert drift["missing_relative_paths"] == ["missing.npy"]
    assert drift["changed_relative_paths"] == ["changed.npy"]
    assert drift["added_relative_paths"] == ["added.npy"]
    assert drift["added_file_count"] == 1
    assert drift["added_bytes"] == 5


def test_cache_clone_copies_only_declared_read_only_seeds(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "nested").mkdir()
    (source / "nested" / "seed.npy").write_bytes(b"seed")
    rows, _ = tree_inventory(source)
    (source / "undeclared.npy").write_bytes(b"probe output")
    (source / "nested" / "seed.npy").chmod(0o440)

    destination = tmp_path / "destination"
    result = clone_declared_files(source, destination, rows)

    assert result["seed_file_count"] == 1
    assert (destination / "nested" / "seed.npy").read_bytes() == b"seed"
    assert not (destination / "undeclared.npy").exists()
    mode = (destination / "nested" / "seed.npy").stat().st_mode
    assert mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH) == 0


def test_prelaunch_uses_private_cache_overlays_and_debug_materialization() -> None:
    wrapper = (ROOT / "clariden/materialize_phase_cache_debug.sbatch").read_text(encoding="utf-8")
    freeze = (ROOT / "clariden/freeze_prelaunch_benchmark_contract_debug.sbatch").read_text(encoding="utf-8")
    preflight = (ROOT / "scripts/verify_prelaunch_benchmark_contract.py").read_text(encoding="utf-8")
    profile = (ROOT / "scripts/finalize_profile_benchmark.py").read_text(encoding="utf-8")
    pilot = (ROOT / "scripts/finalize_lr_pilot_arm.py").read_text(encoding="utf-8")

    assert "#SBATCH --partition=debug" in wrapper
    assert '"${SLURM_JOB_PARTITION:-}" == debug' in wrapper
    assert "H2G_PHASE1_CACHE_OVERLAY_RECEIPT" in freeze
    assert "H2G_PHASE2_CACHE_OVERLAY_RECEIPT" in freeze
    assert "require_pristine=True" in preflight
    assert "qualification_cache_overlay_audits" in profile
    assert "qualification_cache_overlay_audits" in pilot
