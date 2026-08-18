#!/usr/bin/env python3
"""Small no-network tests for the experiment-owned publication adapters."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).parent


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class PrivateD0StageTests(unittest.TestCase):
    def test_prepare_and_verify_hardlinked_stage(self) -> None:
        module = load("prepare_full8_d0_private_stage")
        with tempfile.TemporaryDirectory() as temporary:
            source, output = Path(temporary) / "source", Path(temporary) / "out"
            files: dict[str, Path] = {}
            def add(relative: str, content: bytes = b"x") -> Path:
                path = source / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
                files[relative] = path
                return path
            pools = {}
            manifests = []
            for index, (pool, tokens) in enumerate(module.EXPECTED_ACTIVE.items()):
                output_data = add(f"megatron/{pool}/bucket.bin", f"{pool}-data".encode())
                catalog = add(f"sequences/{pool}.u64", f"{pool}-catalog".encode())
                manifest_path = source / f"megatron/{pool}/manifest.json"
                write_json(manifest_path, {"outputs": {"data": {"path": str(output_data), "bytes": output_data.stat().st_size, "sha256": module.sha256_file(output_data)}}})
                files[str(manifest_path.relative_to(source))] = manifest_path
                manifests.append({"bucket": index, "manifest_path": str(manifest_path), "manifest_sha256": module.sha256_file(manifest_path), "pool": pool, "task_index": index})
                pools[pool] = {"active_tokens": tokens, "sequence_catalog": {"path": str(catalog), "bytes": catalog.stat().st_size, "sha256": module.sha256_file(catalog)}}
            plan = add("inventory/packing_plan.json", b"{}")
            packed = {"schema_version": "apertus_packed_sequence_corpus_v1", "status": "completed", "global": {"sequence_count": 9}, "packing_plan": {"path": str(plan), "sha256": module.sha256_file(plan)}, "packing_task_manifests": manifests, "pools": pools}
            write_json(source / "inventory/packed_corpus_receipt.json", packed)
            schedule_active, schedule_ids = add("schedules/D0_mixed.active_tokens.u16"), add("schedules/D0_mixed.sequence_ids.u64")
            d0_arm = {"arm_id": "D0_mixed", "pool_active_tokens": {"H": module.EXPECTED_ACTIVE["hplt_new_greek"], "G": module.EXPECTED_ACTIVE["non_hplt_new_greek"], "F": module.EXPECTED_ACTIVE["foreign_replay"], "O": module.EXPECTED_ACTIVE["old_greek_replay"]}, "active_tokens": {"path": str(schedule_active), "bytes": 1, "sha256": module.sha256_file(schedule_active)}, "sequence_ids": {"path": str(schedule_ids), "bytes": 1, "sha256": module.sha256_file(schedule_ids)}}
            schedule = {"status": "completed", "arms": [d0_arm, {**d0_arm, "arm_id": "D1_hard_h_to_g"}]}
            write_json(source / "schedules/schedule_manifest.json", schedule)
            for relative in ("contracts/recipe_8b_full_mixed.sanitized.json", "contracts/execution_profiles.sanitized.json", "evidence/selected_training_content/selected_training_content_receipt.json", "evidence/selected_training_content/selected_training_content.sorted.unique.sha32", "evidence/selected_training_content/foreign_replay.selected.sha32", "evidence/selected_training_content/old_greek_replay.selected.sha32", "inventory/raw/modern.content57", "inventory/catalog/hplt_new_greek.source_local_selected.catalog45", "inventory/catalog/non_hplt_new_greek.source_local_selected.catalog45", "inventory/catalog/foreign_replay.source_local_selected.catalog45", "inventory/catalog/old_greek_replay.source_local_selected.catalog45"):
                add(relative)
            prepared = module.prepare(SimpleNamespace(source_stage=source, output_stage=output))
            self.assertEqual(prepared["status"], "prepared_unverified_payload_hashes")
            self.assertEqual((source / "inventory/packing_plan.json").stat().st_ino, (output / "inventory/packing_plan.json").stat().st_ino)
            receipt_path = output / "payload_sha256_verification.json"
            verified = module.verify(SimpleNamespace(output_stage=output, output=receipt_path))
            self.assertEqual(verified["status"], "passed")
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "verified_payload_hashes")
            self.assertTrue(all(isinstance(row["sha256"], str) and len(row["sha256"]) == 64 for row in manifest["upload_payload_inventory"]))


class PrivateModelPlanTests(unittest.TestCase):
    def test_all_trajectory_branches_are_contract_bound(self) -> None:
        module = load("prepare_private_model_branches")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_root, planned = root / "run", root / "plan"
            init = root / "init.json"
            write_json(init, {"schema_version": "production_polytonic_td_init_verification_v1", "status": "passed"})
            tokenizer = b'{"model":{"vocab":["a"]}}'
            for iteration, attempt, _ in module.CHECKPOINTS:
                attempt_root = run_root / "checkpoint_evaluations" / module.label(iteration) / f"attempt_{attempt}"
                hf = attempt_root / "export/hf"
                hf.mkdir(parents=True)
                config = {**module.GEOMETRY, "tie_word_embeddings": False}
                (hf / "config.json").write_text(json.dumps(config), encoding="utf-8")
                (hf / "tokenizer.json").write_bytes(tokenizer)
                files = [{"relative_path": "config.json", "bytes": (hf / "config.json").stat().st_size, "sha256": module.sha256_file(hf / "config.json")}, {"relative_path": "tokenizer.json", "bytes": len(tokenizer), "sha256": module.sha256_file(hf / "tokenizer.json")}]
                export = {"schema_version": "native_greekmmlu_exact_checkpoint_export_v1", "status": "completed", "model_scale": "8B", "source": {"iteration": iteration}, "hf_export": {"path": str(hf), "tokenizer_json_sha256": module.sha256_file(hf / "tokenizer.json"), "files": files, "tree_manifest_sha256": "f" * 64}}
                export_path = attempt_root / "export/checkpoint_eval_export_receipt.json"
                write_json(export_path, export)
                greek = {"schema_version": "exact_checkpoint_native_greekmmlu_receipt_v1", "status": "completed", "checkpoint": {"iteration": iteration, "export_receipt_path": str(export_path), "export_receipt_sha256": module.sha256_file(export_path)}, "metrics": {"decontaminated": {"n": 16159, "accuracy": 0.5, "choice_nll": 1.0, "correct_answer_bpb": 0.2}}}
                write_json(attempt_root / "exact_checkpoint_native_greekmmlu_receipt.json", greek)
            result = module.prepare(SimpleNamespace(run_root=run_root, initialization_receipt=init, output_root=planned, model_repo="fffoivos/apertus-8b-greek-cpt", workers=8))
            self.assertEqual(len(result["branches"]), 18)
            contract = json.loads((planned / "contracts/step9536-tokens40B.json").read_text())
            self.assertTrue(contract["repository"]["private"])
            self.assertEqual(contract["tokenizer"]["sha256"], hashlib.sha256(tokenizer).hexdigest())


class PublicIdentityTests(unittest.TestCase):
    def test_identity_and_content57_selection(self) -> None:
        module = load("export_public_modern_greek_train")
        with tempfile.TemporaryDirectory() as temporary:
            stage = Path(temporary)
            original_docs = module.EXPECTED_DOCUMENTS
            try:
                module.EXPECTED_DOCUMENTS = {"hplt_new_greek": 1, "non_hplt_new_greek": 1}
                expected = []
                for pool, doc_id, text in (("hplt_new_greek", "one", "άλφα"), ("non_hplt_new_greek", "two", "βήτα")):
                    identity, content = module.source_identity(doc_id, text)
                    expected.append((pool, identity, content))
                    catalog = stage / "inventory/catalog" / f"{pool}.source_local_selected.catalog45"
                    catalog.parent.mkdir(parents=True, exist_ok=True)
                    catalog.write_bytes(b"\0" * 13 + identity + b"\0" * 16)
                content_path = stage / "inventory/raw/modern.content57"
                content_path.parent.mkdir(parents=True, exist_ok=True)
                content_path.write_bytes(b"".join(content + b"\0" + (0).to_bytes(8, "little") + identity for _, identity, content in expected))
                selected = module.selected_expected(stage)
                self.assertEqual(len(selected), 2)
                self.assertEqual(selected[expected[0][1]], (expected[0][2], "hplt_new_greek"))
            finally:
                module.EXPECTED_DOCUMENTS = original_docs


class HubInventoryTests(unittest.TestCase):
    def test_lfs_and_small_file_are_both_checked(self) -> None:
        module = load("verify_hub_branch_inventory")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            small = root / "README.md"
            small.write_text("ok", encoding="utf-8")
            large_sha = hashlib.sha256(b"large").hexdigest()
            siblings = [SimpleNamespace(rfilename="README.md", size=2, lfs=None), SimpleNamespace(rfilename="model.safetensors", size=5, lfs=SimpleNamespace(oid=large_sha)), SimpleNamespace(rfilename=".gitattributes", size=1, lfs=None)]
            checked = module.inventory_check([{"relative_path": "README.md", "bytes": 2, "sha256": module.sha256_file(small)}, {"relative_path": "model.safetensors", "bytes": 5, "sha256": large_sha}], siblings, lambda name: small if name == "README.md" else (_ for _ in ()).throw(AssertionError(name)))
            self.assertEqual([row["method"] for row in checked], ["downloaded_content_sha256", "hub_lfs_oid", "hub_generated_lfs_routing_metadata"])


class DatasetManifestTests(unittest.TestCase):
    def test_manifest_paths_reject_drift(self) -> None:
        module = load("upload_frozen_dataset")
        with tempfile.TemporaryDirectory() as temporary:
            stage = Path(temporary)
            payload = stage / "README.md"
            payload.write_text("ok", encoding="utf-8")
            paths = module.manifest_paths(stage, {"upload_payload_inventory": [{"relative_path": "README.md", "bytes": 2, "sha256": module.sha256_file(payload)}]})
            self.assertEqual(paths, ["README.md", "manifest.json"])
            with self.assertRaises(ValueError):
                module.manifest_paths(stage, {"upload_payload_inventory": [{"relative_path": "../unsafe", "bytes": 0, "sha256": "0" * 64}]})

    def test_private_receipt_must_exactly_bind_upload_inventory(self) -> None:
        module = load("upload_frozen_dataset")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = root / "payload.bin"
            payload.write_bytes(b"x")
            receipt = root / "verified.json"
            row = {"relative_path": "payload.bin", "bytes": 1, "sha256": hashlib.sha256(b"x").hexdigest()}
            write_json(receipt, {"schema_version": "apertus_full8_d0_private_payload_hash_verification_v1", "status": "passed", "files": [row]})
            manifest = {"upload_payload_inventory": [row], "hash_verification": {"receipt": str(receipt), "sha256": module.sha256_file(receipt)}}
            module.verify_private_payload_receipt(manifest)
            manifest["upload_payload_inventory"] = [{**row, "sha256": "0" * 64}]
            with self.assertRaises(ValueError):
                module.verify_private_payload_receipt(manifest)


if __name__ == "__main__":
    unittest.main()
