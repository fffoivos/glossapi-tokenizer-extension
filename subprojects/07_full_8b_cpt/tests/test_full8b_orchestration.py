from __future__ import annotations

import json
import re
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def training_log(path: Path, milliseconds: float, *, start: int = 1, end: int = 288) -> None:
    rows = []
    for iteration in range(start, end + 1):
        rows.append(
            f"iteration {iteration}/ 19248 | elapsed time per iteration (ms): {milliseconds:.3f} | "
            "lm loss: 6.000000 | grad norm: 0.500000 | params norm: 100.000000 | "
            "number of skipped iterations: 0 | number of nan iterations: 0"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


class Full8BOrchestrationTests(unittest.TestCase):
    def test_every_code_root_literal_exists_in_the_frozen_repository(self) -> None:
        repository_root = ROOT.parents[1]
        pattern = re.compile(r'\$FULL8_CODE_ROOT/([^"\\\s]+)')
        missing = []
        for script in sorted((ROOT / "clariden").iterdir()):
            if not script.is_file():
                continue
            for relative in pattern.findall(script.read_text(encoding="utf-8")):
                if not (repository_root / relative).exists():
                    missing.append(f"{script.name}: {relative}")
        self.assertEqual(missing, [])

    def test_modern_campaign_modules_never_use_system_python(self) -> None:
        unsafe = re.compile(
            r'^(?:exec\s+)?python3\s+"\$FULL8_CODE_ROOT/(?:subprojects/07_full_8b_cpt|subprojects/06_dataset_scheduling_experiments)/[^"\s]+\.py"',
            re.MULTILINE,
        )
        violations = []
        for script in sorted((ROOT / "clariden").iterdir()):
            if script.is_file() and unsafe.search(script.read_text(encoding="utf-8")):
                violations.append(script.name)
        self.assertEqual(violations, [])

    def test_benchmark_restart_loads_iteration_160_not_the_terminal_288(self) -> None:
        submit = (ROOT / "clariden/submit_parallelism_benchmark.sh").read_text()
        train = (ROOT / "clariden/train_segment.sbatch").read_text()
        self.assertEqual(submit.count("FULL8_EXACT_LOAD_ITERATION=160"), 2)
        self.assertIn('initial_common="$common,FULL8_BENCHMARK_SAVE_ITERATIONS=160"', submit)
        self.assertNotIn('FULL8_BENCHMARK_SAVE_ITERATIONS=160,FULL8_BENCHMARK_SAVE_ITERATIONS=', submit)
        self.assertIn("benchmark_load_views", train)
        self.assertIn("latest_checkpointed_iteration.txt", train)

    def test_megatron_cache_is_writable_run_state_not_frozen_dataset_state(self) -> None:
        train = (ROOT / "clariden/train_segment.sbatch").read_text()
        self.assertIn('DATA_CACHE="$FULL8_RUN_ROOT/dataset_cache"', train)
        self.assertIn('--data-cache-path "$DATA_CACHE"', train)
        self.assertNotIn('--data-cache-path "$FULL8_STAGE_ROOT', train)

    def test_profiles_preserve_global_batch(self) -> None:
        for profile in ("dp32_16node", "dp64_32node"):
            subprocess.run(
                [
                    "python3", str(ROOT / "scripts/validate_execution_profile.py"),
                    "--recipe", str(ROOT / "configs/recipe_8b_full_mixed.json"),
                    "--profiles", str(ROOT / "configs/execution_profiles.json"),
                    "--profile-id", profile,
                ], check=True, capture_output=True, text=True,
            )

    def test_owner_decisions_record_risk_without_legal_claim(self) -> None:
        subprocess.run(
            [
                "python3", str(ROOT / "scripts/validate_owner_decisions.py"),
                "--decisions", str(ROOT / "configs/owner_decisions_20260805.json"),
                "--recipe-id", "full8b-mixed-79-20-1-wsd10-v1",
            ], check=True, capture_output=True, text=True,
        )

    def test_parallelism_benchmark_promotes_only_passing_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            control = root / "control"; candidate = root / "candidate"
            for target, ms, elapsed, profile in (
                (control, 8500.0, 2600, "dp32_16node"),
                (candidate, 5000.0, 1440, "dp64_32node"),
            ):
                training_log(target / "segments/updates_0_288/training.log", ms)
                training_log(target / "segments/updates_160_161/training.log", ms, start=161, end=161)
                write(target / "segments/updates_0_288/training_job_receipt.json", {
                    "schema_version":"apertus_full_8b_training_job_v1", "status":"completed",
                    "elapsed_seconds":elapsed, "scientific_digest":"same", "profile_id":profile,
                })
            contract = root / "contract.json"
            write(contract, {
                "schema_version":"apertus_full_8b_parallelism_benchmark_contract_v1", "status":"frozen", "updates":288,
                "sequence_ids":{"prefix_sha256":"abc"}, "goldfish":{"implementation":{"sha256":"def"}},
            })
            output = root / "promotion.json"
            subprocess.run([
                "python3", str(ROOT / "scripts/finalize_parallelism_benchmark.py"),
                "--profiles", str(ROOT / "configs/execution_profiles.json"), "--benchmark-contract", str(contract),
                "--control-root", str(control), "--candidate-root", str(candidate), "--output", str(output),
            ], check=True, capture_output=True, text=True)
            result = json.loads(output.read_text())
            self.assertTrue(result["candidate_promoted"])
            self.assertEqual(result["selected_profile"], "dp64_32node")
            self.assertTrue(all(result["checks"].values()))

    def test_training_attempt_requires_all_thirteen_panels(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            log = root / "train.log"
            training_log(log, 5000.0, start=1, end=25)
            panels = [{"name": f"panel_{index}"} for index in range(13)]
            with log.open("a", encoding="utf-8") as handle:
                for row in panels:
                    handle.write(f"validation loss at iteration 25 on [{row['name']}] | lm loss: 1.0\n")
            manifest = root / "validation.json"; write(manifest, {"panels":panels})
            output = root / "audit.json"
            subprocess.run([
                "python3", str(ROOT / "train/audit_training_attempt.py"), "--log", str(log),
                "--validation-manifest", str(manifest), "--start", "0", "--end", "25", "--output", str(output),
            ], check=True, capture_output=True, text=True)
            self.assertEqual(json.loads(output.read_text())["status"], "passed")

    def test_production_submit_is_receipt_gated_and_dry_run_safe(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selected = root / "selected.json"; gate = root / "gate.json"
            write(selected, {"schema_version":"apertus_full_8b_selected_execution_profile_v1", "status":"frozen", "selection":{"profile_id":"dp64_32node","nodes":32,"segment_boundaries":[0,6416,12832,19248]}})
            write(gate, {"schema_version":"apertus_full_8b_launch_gate_v1", "status":"passed"})
            env = {
                "FULL8_CODE_ROOT":str(ROOT.parents[1]), "FULL8_STAGE_ROOT":"/stage", "FULL8_RUN_ROOT":"/new-run",
                "FULL8_INITIAL_MEGATRON":"/init", "FULL8_PRELAUNCH_ROOT":"/prelaunch",
                "FULL8_SELECTED_PROFILE":str(selected), "FULL8_LAUNCH_GATE":str(gate), "DRY_RUN":"1",
            }
            result = subprocess.run([str(ROOT / "clariden/submit_production.sh")], env={**__import__("os").environ, **env}, text=True, capture_output=True, check=True)
            self.assertIn("dp64_32node", result.stdout)
            self.assertIn("--nodes=32", result.stderr)


if __name__ == "__main__":
    unittest.main()
