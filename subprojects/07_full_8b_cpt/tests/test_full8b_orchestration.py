from __future__ import annotations

import json
import hashlib
import importlib.util
import io
import re
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def training_log(
    path: Path,
    milliseconds: float,
    *,
    start: int = 1,
    end: int = 288,
    timestamps: bool = False,
) -> None:
    rows = []
    epoch = datetime(2026, 8, 5, 12, 0, 0)
    for iteration in range(start, end + 1):
        wall = ""
        if timestamps:
            finished = epoch + timedelta(milliseconds=milliseconds * iteration)
            wall = f"[{finished:%Y-%m-%d %H:%M:%S}] "
        rows.append(
            f"{wall}iteration {iteration}/ 19248 | elapsed time per iteration (ms): {milliseconds:.3f} | "
            "lm loss: 6.000000 | grad norm: 0.500000 | params norm: 100.000000 | "
            "number of skipped iterations: 0 | number of nan iterations: 0"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def benchmark_receipts(root: Path, *, elapsed: float, profile: str) -> None:
    common = {
        "schema_version": "apertus_full_8b_training_job_v1",
        "status": "completed",
        "scientific_digest": "same",
        "profile_id": profile,
    }
    write(root / "segments/updates_0_288/training_job_receipt.json", {
        **common, "elapsed_seconds": elapsed, "start_iteration": 0, "end_iteration": 288,
    })
    write(root / "segments/updates_160_161/training_job_receipt.json", {
        **common, "elapsed_seconds": 1, "start_iteration": 160, "end_iteration": 161,
    })
    checkpoint = root / "checkpoints/iter_0000160"
    checkpoint.mkdir(parents=True)
    view = root / "benchmark_load_views" / f"iter_0000160_for_{profile}"
    view.mkdir(parents=True)
    (view / "latest_checkpointed_iteration.txt").write_text("160\n")
    (view / "iter_0000160").symlink_to(checkpoint)


class Full8BOrchestrationTests(unittest.TestCase):
    def test_selected_content_reader_derives_source_local_order(self) -> None:
        path = ROOT / "dataset/freeze_selected_training_content.py"
        spec = importlib.util.spec_from_file_location("full8_selected_content", path)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tasks = {}
            rows = []
            for task_index, text in enumerate(("first", "second")):
                digest = hashlib.sha256(text.encode()).digest()
                doc_id = f"doc-{task_index}"
                identity = hashlib.sha256(doc_id.encode() + b"\0" + digest).digest()[:16]
                ledger = root / f"{task_index}.jsonl"
                ledger.write_text(json.dumps({
                    "doc_id": doc_id, "text_sha256": digest.hex(), "tokens": task_index + 3,
                }) + "\n", encoding="utf-8")
                manifest = root / f"{task_index}.manifest.json"
                write(manifest, {"outputs": {"retained_ledger": {
                    "path": str(ledger), "rows": 1,
                }}})
                tasks[task_index] = {
                    "pool": "foreign_replay",
                    "source_manifest": {"path": str(manifest)},
                }
                rows.append((2, task_index, 0, task_index + 3, identity, bytes([task_index + 1]) * 16))
            catalog = root / "selected.catalog45"
            array = module.np.asarray(list(reversed(rows)), dtype=module.CATALOG_DTYPE)
            array.tofile(catalog)
            output = io.BytesIO()
            documents, tokens, bindings = module.extract_selected_hashes(
                "foreign_replay", catalog, tasks, output,
            )
            self.assertEqual((documents, tokens, len(bindings)), (2, 7, 2))
            self.assertEqual(output.getvalue(), b"".join(
                hashlib.sha256(text.encode()).digest() for text in ("first", "second")
            ))

    def test_sha256_sortedness_check_uses_exact_raw_byte_order(self) -> None:
        path = ROOT / "dataset/build_clean_replay_validation.py"
        spec = importlib.util.spec_from_file_location("full8_clean_replay", path)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        values = module.np.asarray([
            b"\x00" * 31 + b"\x01",
            b"\x00" * 30 + b"\x01\x00",
            b"\x01" + b"\x00" * 31,
        ], dtype="V32")
        module.require_strictly_sorted_sha256(values, chunk_rows=2)
        with self.assertRaisesRegex(ValueError, "not unique and sorted"):
            module.require_strictly_sorted_sha256(values[[1, 0, 2]], chunk_rows=2)
        with self.assertRaisesRegex(ValueError, "not unique and sorted"):
            module.require_strictly_sorted_sha256(values[[0, 0, 2]], chunk_rows=2)

    def test_corrected_initial_hf_changes_only_evaluation_geometry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            config = {
                "rope_theta": 12_000_000.0, "max_position_embeddings": 65_536,
                "tie_word_embeddings": False, "vocab_size": 148_992,
                "rope_scaling": {"factor": 8.0, "rope_type": "llama3"},
            }
            (source / "config.json").write_text(json.dumps(config), encoding="utf-8")
            names = {
                "generation_config.json", "model-00001-of-00004.safetensors",
                "model-00002-of-00004.safetensors", "model-00003-of-00004.safetensors",
                "model-00004-of-00004.safetensors", "model.safetensors.index.json",
                "special_tokens_map.json", "tokenizer_config.json", "tokenizer.json",
            }
            for name in names:
                (source / name).write_bytes(name.encode())
            (source / "tokenizer.json").write_text(
                json.dumps({"version": "1.0", "model": {"vocab": {"a": 0}}}),
                encoding="utf-8",
            )
            verification = root / "verification.json"
            write(verification, {
                "standard_max_abs_diff": 0.0, "r17_max_abs_diff": 0.0,
                "xielu_max_abs_diff": 0.0, "qk_norm_max_abs_diff": 0.0,
                "orig_only": [], "trip_only": [], "shape_mismatches": [],
                "standard_changed_over_tol_count": 0, "r17_changed_over_tol_count": 0,
                "logits": {"logit_max_abs_diff": 0.0},
            })
            output = root / "model"
            receipt = root / "receipt.json"
            canonical_tokenizer = root / "canonical-tokenizer.json"
            canonical_tokenizer.write_text(json.dumps(
                json.loads((source / "tokenizer.json").read_text()), indent=2,
            ) + "\n", encoding="utf-8")
            subprocess.run([
                "python3", str(ROOT / "evaluation/materialize_corrected_initial_hf.py"),
                "--source", str(source), "--roundtrip-verification", str(verification),
                "--expected-verification-sha256", hashlib.sha256(verification.read_bytes()).hexdigest(),
                "--expected-tokenizer-sha256", hashlib.sha256(canonical_tokenizer.read_bytes()).hexdigest(),
                "--canonical-tokenizer-json", str(canonical_tokenizer),
                "--output-root", str(output), "--receipt", str(receipt),
            ], check=True, capture_output=True, text=True)
            corrected = json.loads((output / "config.json").read_text())
            self.assertEqual(corrected["rope_theta"], 500_000.0)
            self.assertEqual(corrected["max_position_embeddings"], 4_096)
            self.assertEqual(
                {key for key in config if config[key] != corrected[key]},
                {"rope_theta", "max_position_embeddings"},
            )
            frozen = json.loads(receipt.read_text())
            self.assertTrue(frozen["zero_tensor_and_logit_drift"])
            self.assertEqual(
                hashlib.sha256((output / "tokenizer.json").read_bytes()).hexdigest(),
                hashlib.sha256(canonical_tokenizer.read_bytes()).hexdigest(),
            )
            self.assertEqual(frozen["file_count"], 10)

    def test_graceful_finalizer_uses_resume_record_after_tracker_advances(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            interrupted = root / "segments/updates_0_288"
            restarted = root / "segments/updates_11_12"
            training_log(interrupted / "training.log", 8500.0, start=1, end=11)
            training_log(restarted / "training.log", 8500.0, start=12, end=12)
            (interrupted / "graceful_stop_requested").touch()
            write(interrupted / "training_job_receipt.json", {
                "schema_version": "apertus_full_8b_training_job_v1",
                "status": "gracefully_stopped", "profile_id": "dp32_16node",
                "requested_end_iteration": 288, "last_logged_iteration": 11,
            })
            write(restarted / "training_job_receipt.json", {
                "schema_version": "apertus_full_8b_training_job_v1",
                "status": "completed", "profile_id": "dp32_16node",
                "start_iteration": 11, "end_iteration": 12,
            })
            checkpoint = root / "checkpoints/iter_0000011"
            checkpoint.mkdir(parents=True)
            (checkpoint / ".metadata").write_bytes(b"checkpoint-11")
            (root / "checkpoints/latest_checkpointed_iteration.txt").write_text("12\n")
            write(root / "resume_submission.json", {
                "schema_version": "apertus_full_8b_graceful_resume_submission_v1",
                "status": "submitted", "checkpoint_iteration": 11,
            })
            output = root / "graceful_stop_smoke_receipt.json"
            subprocess.run([
                "python3", str(ROOT / "scripts/finalize_graceful_stop_smoke.py"),
                "--run-root", str(root), "--interrupted-root", str(interrupted),
                "--restart-root", str(restarted), "--output", str(output),
            ], check=True, capture_output=True, text=True)
            receipt = json.loads(output.read_text())
            self.assertEqual(receipt["checkpoint"]["iteration"], 11)
            self.assertEqual(receipt["checkpoint"]["current_tracker_iteration"], 12)

    def test_code_bundle_verifier_rejects_unreceipted_files(self) -> None:
        path = ROOT / "scripts/contract.py"
        spec = importlib.util.spec_from_file_location("full8_contract", path)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "bundle"
            root.mkdir()
            payload = root / "entry.py"
            payload.write_text("value = 1\n", encoding="utf-8")
            row = {
                "relative_path": "entry.py",
                "bytes": payload.stat().st_size,
                "sha256": hashlib.sha256(payload.read_bytes()).hexdigest(),
            }
            receipt = Path(temporary) / "receipt.json"
            write(receipt, {
                "schema_version": "apertus_mini_immutable_code_bundle_v1",
                "status": "frozen",
                "kind": "scientific",
                "root": str(root.resolve()),
                "file_count": 1,
                "tree_sha256": hashlib.sha256(
                    json.dumps([row], separators=(",", ":"), sort_keys=True).encode()
                ).hexdigest(),
                "files": [row],
                "exclusions": {"directory_parts": [], "file_suffixes": []},
            })
            module.verify_code_bundle_receipt(receipt, root)
            (root / "unreceipted.py").write_text("value = 2\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "inventory drift"):
                module.verify_code_bundle_receipt(receipt, root)

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
        self.assertEqual(submit.count("--time=00:20:00"), 2)

    def test_megatron_cache_is_writable_run_state_not_frozen_dataset_state(self) -> None:
        train = (ROOT / "clariden/train_segment.sbatch").read_text()
        self.assertIn('DATA_CACHE="$FULL8_RUN_ROOT/dataset_cache"', train)
        self.assertIn('--data-cache-path "$DATA_CACHE"', train)
        self.assertNotIn('--data-cache-path "$FULL8_STAGE_ROOT', train)

    def test_checkpoint_compatibility_is_dependency_closed_and_preloaded(self) -> None:
        train = (ROOT / "clariden/train_segment.sbatch").read_text()
        shim = (ROOT / "runtime_compat/sitecustomize.py").read_text()
        self.assertIn('RUNTIME_COMPAT="$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/runtime_compat"', train)
        self.assertIn('export PYTHONPATH="$RUNTIME_COMPAT:$MEGATRON:$TRAINING_CODE"', train)
        self.assertIn('np.product = np.prod', shim)
        self.assertIn('_apertus_preserves_dynamic_metadata', shim)

    def test_full_8b_greekmmlu_jobs_have_measured_wall_time_margin(self) -> None:
        initial = (ROOT / "clariden/run_initial_greekmmlu.sbatch").read_text()
        prelaunch = (ROOT / "clariden/submit_conversion_smoke.sh").read_text()
        production = (ROOT / "evaluation/submit_greekmmlu_checkpoint.sh").read_text()
        self.assertIn("#SBATCH --time=01:15:00", initial)
        self.assertIn("--time=01:15:00", prelaunch)
        self.assertIn("--time=01:15:00", production)

    def test_full_8b_conversion_uses_sharded_exact_mapping_without_oom_logit_test(self) -> None:
        shared = (
            ROOT.parent
            / "06_dataset_scheduling_experiments/clariden/convert_checkpoint_for_native_greekmmlu.sbatch"
        ).read_text()
        prelaunch = (ROOT / "clariden/submit_conversion_smoke.sh").read_text()
        production = (ROOT / "evaluation/submit_greekmmlu_checkpoint.sh").read_text()
        verifier = (ROOT / "evaluation/verify_exact_checkpoint_weight_mapping_8b.py").read_text()
        finalizer = (ROOT / "evaluation/finalize_checkpoint_export_8b.py").read_text()
        self.assertIn("EXPORT_MODEL_SCALE=${EXPORT_MODEL_SCALE:-0p5B}", shared)
        self.assertIn("verify_exact_checkpoint_weight_mapping_8b.py", shared)
        self.assertIn("finalize_checkpoint_export_8b.py", shared)
        self.assertIn("EXPORT_MODEL_SCALE=8B", prelaunch)
        self.assertIn("EXPORT_MODEL_SCALE=8B", production)
        self.assertIn('"output_layer.weight"', verifier)
        self.assertIn('"model.safetensors.index.json"', verifier)
        self.assertIn('"source_parameter_tensors_expected": len(expected_source)', verifier)
        self.assertIn('"runtime_logit_diagnostics": "skipped_single_gpu_memory_limit"', finalizer)
        self.assertIn('"parity_acceptance_path": "bit_exact_parameter_mapping"', finalizer)

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
                training_log(target / "segments/updates_0_288/training.log", ms, timestamps=True)
                training_log(target / "segments/updates_160_161/training.log", ms, start=161, end=161)
                benchmark_receipts(target, elapsed=elapsed, profile=profile)
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

    def test_benchmark_fixed_startup_is_amortized_over_production_segment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            control = root / "control"; candidate = root / "candidate"
            for target, ms, elapsed, profile in (
                (control, 8500.0, 288 * 8.5 + 160, "dp32_16node"),
                (candidate, 5000.0, 288 * 5.0 + 160, "dp64_32node"),
            ):
                training_log(target / "segments/updates_0_288/training.log", ms, timestamps=True)
                training_log(target / "segments/updates_160_161/training.log", ms, start=161, end=161)
                benchmark_receipts(target, elapsed=elapsed, profile=profile)
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
            performance = json.loads(output.read_text())["performance"]
            self.assertGreater(performance["candidate_benchmark_wall_seconds_per_update"], 5.5)
            self.assertLess(performance["candidate_projected_production_wall_seconds_per_update"], 5.1)

    def test_benchmark_never_falls_back_to_a_control_with_failed_restart(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            control = root / "control"; candidate = root / "candidate"
            for target, ms, elapsed, profile in (
                (control, 8500.0, 2600, "dp32_16node"),
                (candidate, 5000.0, 1440, "dp64_32node"),
            ):
                training_log(target / "segments/updates_0_288/training.log", ms, timestamps=True)
                training_log(target / "segments/updates_160_161/training.log", ms, start=161, end=161)
                benchmark_receipts(target, elapsed=elapsed, profile=profile)
            restart = control / "segments/updates_160_161/training.log"
            restart.write_text(restart.read_text().replace("lm loss: 6.000000", "lm loss: 6.100000"))
            contract = root / "contract.json"
            write(contract, {
                "schema_version":"apertus_full_8b_parallelism_benchmark_contract_v1", "status":"frozen", "updates":288,
                "sequence_ids":{"prefix_sha256":"abc"}, "goldfish":{"implementation":{"sha256":"def"}},
            })
            output = root / "promotion.json"
            result = subprocess.run([
                "python3", str(ROOT / "scripts/finalize_parallelism_benchmark.py"),
                "--profiles", str(ROOT / "configs/execution_profiles.json"), "--benchmark-contract", str(contract),
                "--control-root", str(control), "--candidate-root", str(candidate), "--output", str(output),
            ], capture_output=True, text=True)
            receipt = json.loads(output.read_text())
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(receipt["status"], "failed")
            self.assertIsNone(receipt["selected_profile"])
            self.assertFalse(receipt["fallback_control_viable"])

    def test_cross_node_restart_allows_bounded_gradient_reduction_roundoff(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            control = root / "control"; candidate = root / "candidate"
            for target, ms, elapsed, profile in (
                (control, 8500.0, 2600, "dp32_16node"),
                (candidate, 5000.0, 1440, "dp64_32node"),
            ):
                training_log(target / "segments/updates_0_288/training.log", ms, timestamps=True)
                training_log(target / "segments/updates_160_161/training.log", ms, start=161, end=161)
                benchmark_receipts(target, elapsed=elapsed, profile=profile)
            restart = control / "segments/updates_160_161/training.log"
            restart.write_text(restart.read_text().replace("grad norm: 0.500000", "grad norm: 0.505000"))
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
            receipt = json.loads(output.read_text())
            self.assertTrue(receipt["checks"]["control_restart_provenance"])
            self.assertTrue(receipt["checks"]["control_restart_numerically_equivalent"])
            self.assertTrue(receipt["restart"]["control"]["numerical"]["gradient_norm"]["within_tolerance"])
            self.assertEqual(receipt["restart"]["control"]["numerical"]["exact_logged_fields"], {"loss": True, "params": True})

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
            write(gate, {"schema_version":"apertus_full_8b_launch_gate_v1", "status":"passed", "initialization_checkpoint":{"root":"/init"}})
            env = {
                "FULL8_CODE_ROOT":str(ROOT.parents[1]), "FULL8_CODE_BUNDLE_RECEIPT":"/bundle.json",
                "FULL8_STAGE_ROOT":"/stage", "FULL8_RUN_ROOT":"/new-run",
                "FULL8_INITIAL_MEGATRON":"/init", "FULL8_PRELAUNCH_ROOT":"/prelaunch",
                "FULL8_SELECTED_PROFILE":str(selected), "FULL8_LAUNCH_GATE":str(gate), "DRY_RUN":"1",
            }
            result = subprocess.run([str(ROOT / "clariden/submit_production.sh")], env={**__import__("os").environ, **env}, text=True, capture_output=True, check=True)
            self.assertIn("dp64_32node", result.stdout)
            self.assertIn("--nodes=32", result.stderr)

    def test_graceful_stop_is_batch_signalled_pollable_and_retry_safe(self) -> None:
        train = (ROOT / "clariden/train_segment.sbatch").read_text()
        self.assertIn("#SBATCH --signal=B:USR1@600", train)
        self.assertNotIn("SIGUSR2", train)
        self.assertIn("trap request_graceful_stop USR1 TERM INT", train)
        self.assertIn('mv "$FULL8_RUN_ROOT/triggers/$trigger"', train)
        self.assertIn("TRAIN_PIPELINE_PID=$!", train)
        self.assertIn('while kill -0 "$TRAIN_PIPELINE_PID"', train)

    def test_supervisor_recovers_a_signalled_incomplete_attempt(self) -> None:
        path = ROOT / "scripts/supervise_campaign.py"
        sys.path.insert(0, str(path.parent))
        spec = importlib.util.spec_from_file_location("full8_supervisor", path)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            log = root / "training.log"
            training_log(log, 5000.0, start=1, end=25)
            marker = root / "graceful_stop_requested"
            marker.touch()
            self.assertEqual(module.last_logged_iteration(log), 25)
            self.assertTrue(module.retryable_terminal("FAILED", marker, 25, 100))
            marker.unlink()
            self.assertFalse(module.retryable_terminal("FAILED", marker, 25, 100))
            self.assertTrue(module.retryable_terminal("TIMEOUT", marker, 25, 100))

    def test_initial_greekmmlu_finalizer_rejects_wrong_rope_geometry_first(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "model"; model.mkdir()
            write(model / "config.json", {"rope_theta": 12_000_000, "max_position_embeddings": 65_536})
            result = subprocess.run([
                "python3", str(ROOT / "evaluation/finalize_hf_greekmmlu.py"),
                "--model", str(model), "--evaluation-root", str(root / "missing"),
                "--model-label", "bad", "--clean-subset-manifest", str(root / "missing.json"),
                "--output", str(root / "receipt.json"),
            ], text=True, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("HF evaluation geometry drift", result.stderr)

    def test_launch_gate_is_computed_not_stamped_true(self) -> None:
        gate = (ROOT / "scripts/build_launch_gate.py").read_text()
        self.assertNotIn("{name: True for name in recipe", gate)
        self.assertIn("verify_code_bundle_receipt", gate)
        self.assertIn("initial checkpoint inventory drift", gate)
        self.assertIn("initial validation/manifest drift", gate)

    def test_final_launch_handoff_is_dependency_safe_and_fail_closed(self) -> None:
        handoff = (ROOT / "clariden/finalize_and_submit_production.sbatch").read_text()
        environment = (ROOT / "scripts/capture_launch_environment.py").read_text()
        self.assertIn('FULL8_LAUNCH_AUTHORIZATION:?set explicit production authorization', handoff)
        self.assertIn('== APERTUS8B_FULL_MIXED_CPT', handoff)
        self.assertLess(handoff.index("capture_launch_environment.py"), handoff.index("build_launch_gate.py"))
        self.assertLess(handoff.index("build_launch_gate.py"), handoff.index("DRY_RUN=1"))
        self.assertLess(handoff.index("DRY_RUN=1"), handoff.index("DRY_RUN=0"))
        self.assertIn('[[ ! -e "$environment" && ! -e "$gate" && ! -e "$FULL8_RUN_ROOT" ]]', handoff)
        self.assertIn("--conversion-smoke", handoff)
        self.assertIn("--benchmark-receipt", handoff)
        self.assertIn("--initial-hf-receipt", handoff)
        self.assertIn('"--uenv-passthrough=ignore"', environment)


if __name__ == "__main__":
    unittest.main()
