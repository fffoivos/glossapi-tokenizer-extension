from __future__ import annotations

import copy
import importlib.util
import json
import math
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "evaluation"))

from freeze_owner_authorization import explicit_run_authorization  # noqa: E402
from contract_utils import read_json  # noqa: E402
from build_retok_reference_init import load_retok  # noqa: E402
from freeze_hard_h_to_g_contract import (  # noqa: E402
    ARTIFACTS_BY_STAGE,
    CHECKPOINT_UPDATES,
    DATA_PIPELINE,
    PRE_EXTENSION_ARTIFACTS,
    PRE_FINALIZATION_ARTIFACTS,
    PRE_MAIN_ARTIFACTS,
    PRE_MAIN_ARTIFACTS_BY_SCALE,
    PRE_MAIN_SHARED_ARTIFACTS,
    PRE_MAIN_TERMINAL_ARTIFACTS,
    PRE_SECOND_EXTENSION_ARTIFACTS,
    REQUIRED_ARTIFACTS,
    ROLE_SCHEMAS,
    artifacts_for_stage,
    role_semantics_match,
    validate_allocation,
    validate_artifact_manifest,
    validate_experiment,
)
from build_greekmmlu_sentinels import hamilton, select_nested, stable_digest  # noqa: E402
from validate_greekmmlu_sentinels import choice_nll  # noqa: E402
from inventory_hard_h_to_g_assets import decimal_product_equal, inspect_asset  # noqa: E402
from freeze_legacy_public_evaluator import validate_contract as validate_legacy_contract  # noqa: E402
from freeze_statistical_decision_contract import validate_statistics  # noqa: E402
from phase_local_data_index_guard import (  # noqa: E402
    override_train_dataset_samples,
    validate_contract as validate_phase_cursor,
)
from build_phase3_unseen_catalog import (  # noqa: E402
    COMPONENT_REQUESTED_SAMPLES,
    SEQUENCE_LENGTH,
    eligible_rows,
    phase2_identities,
)
import prepare_greek_source_stream as prepared_greek  # noqa: E402
from constant_floor_resume import enforce_constant_floor  # noqa: E402
from patch_bakeoff_scale_geometry import (  # noqa: E402
    NEW_METADATA,
    NEW_NETWORK,
    NEW_DATA_THREADS,
    OLD_DATA_THREADS,
    OLD_METADATA,
    OLD_NETWORK,
    patch_trainer,
)
import build_checkpoint_permit as checkpoint_permits  # noqa: E402
import build_training_run_permit as training_run_permits  # noqa: E402
from audit_training_checkpoint import parse_training_log  # noqa: E402
from contract_utils import file_binding, require_relative_inventory, sha256_file  # noqa: E402
from freeze_phase_blend_cache import (  # noqa: E402
    PHASE3_COMPONENT_REQUESTED_SAMPLES,
    require_compatible_code_bundle,
    validate_data_path_spec,
)
from freeze_reused_validation_panels import extract_panel_row  # noqa: E402
from derive_replay_recipe_for_reused_panels import receipt_bound_glob  # noqa: E402
from freeze_tokenized_stream import INDEX_MAGIC, inspect_index, write_document_catalog  # noqa: E402
from export_realized_document_ledger import documents_for_component  # noqa: E402
from finalize_lr_pilot_arm import parse_validation_log  # noqa: E402
from freeze_replay_source_inventory import identity_spec  # noqa: E402
from freeze_pre_main_data_authorities import binding_without_counts, byte_identity  # noqa: E402


MIX_BUILDER_PATH = (
    ROOT.parent / "03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/"
    "init_bakeoff/corpus_build/mix_builder.py"
)
MIX_BUILDER_SPEC = importlib.util.spec_from_file_location("h2g_test_mix_builder", MIX_BUILDER_PATH)
assert MIX_BUILDER_SPEC is not None and MIX_BUILDER_SPEC.loader is not None
mix_builder = importlib.util.module_from_spec(MIX_BUILDER_SPEC)
MIX_BUILDER_SPEC.loader.exec_module(mix_builder)


class HardHToGContractTests(unittest.TestCase):
    def test_owner_launch_language_does_not_require_magic_scope_keywords(self) -> None:
        self.assertTrue(explicit_run_authorization("the goal is run the training dummy"))
        self.assertTrue(explicit_run_authorization("I authorize the 8B run"))
        self.assertFalse(explicit_run_authorization("please continue checking configuration"))
    @classmethod
    def setUpClass(cls) -> None:
        cls.experiment = read_json(ROOT / "configs/hard_h_to_g_replication_v1.json")
        cls.allocation = read_json(ROOT / "configs/hard_h_to_g_allocation_v1.json")

    def test_static_contract_and_geometry(self) -> None:
        derived = validate_experiment(copy.deepcopy(self.experiment))
        self.assertEqual(derived["realized_main_token_slots"], 13_497_270_272)
        self.assertEqual(derived["realized_terminal_token_slots"], 15_493_758_976)
        self.assertEqual(derived["extension_token_slots"], 1_996_488_704)
        self.assertEqual(list(map(int, derived["checkpoint_token_slots"])), CHECKPOINT_UPDATES)
        self.assertEqual(self.experiment["launch"]["required_pre_main_launch_artifacts"], REQUIRED_ARTIFACTS)
        self.assertEqual(self.experiment["data"]["pipeline"], DATA_PIPELINE)
        self.assertLess(DATA_PIPELINE.index("normalize_selected_replay_then_filter_native_suite_then_filter_greekmmlu"), DATA_PIPELINE.index("verify_v2_anonymization_is_stage_b_noop_and_apply_stage_b_only_to_twice_filtered_replay"))
        self.assertLess(DATA_PIPELINE.index("audit_exact_replay_stage_b_bytes_for_zero_greekmmlu_and_native_suite_matches"), DATA_PIPELINE.index("split_final_replay_into_foreign_and_old_greek"))
        self.assertFalse(self.experiment["training"]["cross_document_attention"])
        self.assertTrue(self.experiment["training"]["attention_mask_reset_at_document_boundary"])
        self.assertTrue(self.experiment["training"]["position_reset_at_document_boundary"])
        self.assertTrue(self.experiment["training"]["eod_loss_masking"])
        self.assertEqual(ARTIFACTS_BY_STAGE, {
            "pre_main": PRE_MAIN_ARTIFACTS,
            "pre_extension": PRE_EXTENSION_ARTIFACTS,
            "pre_second_extension": PRE_SECOND_EXTENSION_ARTIFACTS,
            "pre_finalization": PRE_FINALIZATION_ARTIFACTS,
        })
        self.assertNotIn("phase_3_unseen_blend_and_capacity_receipt", PRE_MAIN_ARTIFACTS)
        for scale in ("8b", "1p5b"):
            self.assertEqual(
                artifacts_for_stage("pre_main", scale),
                PRE_MAIN_SHARED_ARTIFACTS
                + PRE_MAIN_ARTIFACTS_BY_SCALE[scale]
                + PRE_MAIN_TERMINAL_ARTIFACTS,
            )
        self.assertNotIn("1p5b_td_init", artifacts_for_stage("pre_main", "8b"))
        self.assertNotIn("8b_init_roundtrip", artifacts_for_stage("pre_main", "1p5b"))
        with self.assertRaisesRegex(ValueError, "requires an exact model scale"):
            artifacts_for_stage("pre_main")

    def test_stage_b_byte_noop_ignores_output_location_but_not_payload(self) -> None:
        source = {"path": "/prepared.jsonl", "bytes": 123, "sha256": "a" * 64}
        relocated = {"path": "/stage_b.jsonl", "bytes": 123, "rows": 7, "sha256": "a" * 64}
        changed = {"path": "/stage_b.jsonl", "bytes": 123, "rows": 7, "sha256": "b" * 64}
        self.assertEqual(byte_identity(source), byte_identity(relocated))
        self.assertNotEqual(byte_identity(source), byte_identity(changed))
        self.assertIn("phase_3_unseen_blend_and_capacity_receipt", PRE_EXTENSION_ARTIFACTS)
        self.assertEqual(set(ROLE_SCHEMAS), {role for roles in ARTIFACTS_BY_STAGE.values() for role in roles})

    def test_receipt_lineage_ignores_only_producer_count_metadata(self) -> None:
        producer_output = {"path": "/clean.jsonl", "bytes": 123, "rows": 7, "sha256": "a" * 64}
        consumer_input = {"path": "/clean.jsonl", "bytes": 123, "sha256": "a" * 64}
        wrong_input = {"path": "/other.jsonl", "bytes": 123, "sha256": "a" * 64}
        self.assertEqual(binding_without_counts(producer_output), consumer_input)
        self.assertNotEqual(binding_without_counts(producer_output), wrong_input)

    def test_phase_cache_adoption_requires_a_preaccepted_bundle_identity(self) -> None:
        producer = {"root": "/immutable/v24", "tree_sha256": "a" * 64}
        require_compatible_code_bundle(producer, {("/immutable/v24", "a" * 64)})
        with self.assertRaisesRegex(ValueError, "phase cache code-bundle drift"):
            require_compatible_code_bundle(producer, {("/immutable/v26", "b" * 64)})

        authority = (ROOT / "scripts/freeze_pre_main_data_authorities.py").read_text(encoding="utf-8")
        self.assertIn("accepted_code_bundles=accepted_cache_code_bundles", authority)
        self.assertIn('require_accepted_producer(value, accepted_producers, f"Phase-{phase} blend cache")', authority)

    def test_artifact_roles_reject_wrong_schema_and_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            wrong = root / "wrong.json"
            wrong.write_text(json.dumps({"schema_version": "irrelevant_passing_receipt_v1", "status": "passed"}), encoding="utf-8")
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({
                "schema_version": "apertus_hard_h_to_g_artifact_manifest_v1",
                "artifacts": {"replay_split": file_binding(wrong)},
            }), encoding="utf-8")
            _, blockers = validate_artifact_manifest(manifest, ["replay_split"])
            self.assertEqual(blockers, ["replay_split:schema_irrelevant_passing_receipt_v1"])

        valid_8b_permit = {
            "schema_version": ROLE_SCHEMAS["8b_training_run_permit"],
            "scale": "8b",
            "profile": {
                "profile_id": "dp32_16node",
                "global_batch_sequences": 1024,
            },
            "learning_rate": {
                "peak": "5.5e-5",
                "floor": "5.5e-6",
                "terminal_ratio": "0.1",
            },
        }
        self.assertTrue(role_semantics_match("8b_training_run_permit", valid_8b_permit))
        wrong_scale = copy.deepcopy(valid_8b_permit)
        wrong_scale["scale"] = "1p5b"
        self.assertFalse(role_semantics_match("8b_training_run_permit", wrong_scale))

        owner = {
            "schema_version": ROLE_SCHEMAS["owner_production_authorization"],
            "authorization_stage": "pre_main",
            "scale": "8b",
        }
        self.assertTrue(role_semantics_match("owner_production_authorization", owner, expected_scale="8b"))
        self.assertFalse(role_semantics_match("owner_production_authorization", owner, expected_scale="1p5b"))

    def test_replay_inventory_uses_a_bounded_finemath_file_row_identity(self) -> None:
        source = identity_spec("replay_t1_english_edu", ["text", "id"], ["id"], "a" * 64)
        self.assertEqual(source, {
            "identity_mode": "source_columns",
            "present_identity_columns": ["id"],
        })
        finemath = identity_spec("math_finemath", ["text"], ["id"], "b" * 64)
        self.assertEqual(finemath["identity_mode"], "immutable_file_sha256_plus_zero_based_row_index")
        self.assertEqual(finemath["synthetic_identity"]["file_sha256"], "b" * 64)
        with self.assertRaisesRegex(ValueError, "no configured identity column"):
            identity_spec("replay_t1_english_edu", ["text"], ["id"], "c" * 64)

    def test_sample_order_drift_fails(self) -> None:
        value = copy.deepcopy(self.experiment)
        value["historical_target"]["curriculum_order_mode"] = "physical_order"
        with self.assertRaisesRegex(ValueError, "sample order drift"):
            validate_experiment(value)

    def test_second_dedup_fails(self) -> None:
        value = copy.deepcopy(self.experiment)
        value["data"]["source_dataset"]["additional_global_deduplication_allowed"] = True
        with self.assertRaisesRegex(ValueError, "second dedup enabled"):
            validate_experiment(value)

    def test_replay_reconstruction_cannot_claim_historical_document_identity(self) -> None:
        value = copy.deepcopy(self.experiment)
        value["data"]["replay_reconstruction"]["historical_document_identity_claimed"] = True
        with self.assertRaisesRegex(ValueError, "falsely claims historical document identity"):
            validate_experiment(value)

    def test_historical_replay_mix_geometry_is_frozen(self) -> None:
        replay = self.experiment["data"]["replay_mix_builder"]
        self.assertEqual(replay["target_tokens"], 5_000_000_000)
        self.assertEqual(replay["source_shard_count"], 16)
        self.assertEqual(replay["target_tokens_per_shard"], 312_500_000)
        self.assertEqual(replay["seed"], 20260611)
        wrapper = (ROOT / "clariden/build_replay_selection_debug.sbatch").read_text(encoding="utf-8")
        self.assertIn("H2G_REPLAY_MIX_SHARDS:-16", wrapper)
        self.assertIn("--seed 20260611", wrapper)
        self.assertIn('--source-shard-count "$mix_shards"', wrapper)

    def test_historical_modern_mix_geometry_is_frozen(self) -> None:
        modern = self.experiment["data"]["modern_mix_builder"]
        self.assertEqual(modern["hplt_target_tokens"], 8_500_000_000)
        self.assertEqual(modern["openarchives_target_tokens"], 3_700_000_000)
        self.assertEqual(modern["source_shard_count"], 16)
        self.assertEqual(modern["hplt_target_tokens_per_shard"], 531_250_000)
        self.assertEqual(modern["openarchives_target_tokens_per_shard"], 231_250_000)
        self.assertEqual(modern["seed"], 20260611)
        self.assertFalse(modern["historical_document_identity_claimed"])
        wrapper = (ROOT / "clariden/build_modern_mix_selection_debug.sbatch").read_text(encoding="utf-8")
        self.assertIn("hplt) target_tokens=8500000000", wrapper)
        self.assertIn("openarchives) target_tokens=3700000000", wrapper)
        self.assertIn("H2G_MODERN_MIX_SHARDS:-16", wrapper)
        self.assertIn("--seed 20260611", wrapper)
        prepare = (ROOT / "clariden/prepare_greek_source_stream_debug.sbatch").read_text(encoding="utf-8")
        self.assertIn("--selected-mix-receipt", prepare)

    def test_mix_builder_preserves_exact_file_and_source_lineage_support(self) -> None:
        builder = MIX_BUILDER_PATH.read_text(encoding="utf-8")
        self.assertIn('local_parquet_files = spec.get("local_parquet_files")', builder)
        self.assertIn('or row.get(doc_key_field)', builder)
        self.assertIn('"_source_release_row_index"', builder)

    def test_source_view_build_full_hashes_release_payloads(self) -> None:
        builder = (ROOT / "scripts/build_hard_h_to_g_source_views.py").read_text(encoding="utf-8")
        wrapper = (ROOT / "clariden/build_hard_h_to_g_source_views_debug.sbatch").read_text(encoding="utf-8")
        self.assertIn('sha256_file(input_path) == row["sha256"]', builder)
        self.assertIn('"all_parquet_sha256_verified_during_build": True', builder)
        self.assertIn("--release-inspection-receipt", wrapper)

    def test_mix_builder_lineage_extension_does_not_change_shard_selection(self) -> None:
        rows = [
            {
                "text": f"document {index}",
                "source_dataset": "HPLT/ell_Grek_ge8_no_mt_clean60",
                "source_doc_id": f"doc-{index}",
                "_source_release_shard": "train.parquet",
                "_source_release_row_index": index,
            }
            for index in range(7)
        ]
        fake_datasets = mock.MagicMock()
        fake_datasets.load_dataset.return_value = rows
        spec = {
            "name": "greek_hplt_70",
            "id": "fixture",
            "text_column": "text",
            "doc_key_field": "source_doc_id",
        }
        with mock.patch.dict(sys.modules, {"datasets": fake_datasets}):
            selected = list(mix_builder._build_source_stream(spec, None, 1, 3))
        self.assertEqual([row["doc_id"] for row in selected], ["doc-1", "doc-4"])
        self.assertEqual([row["text"] for row in selected], ["document 1", "document 4"])
        self.assertEqual([row["_source_release_row_index"] for row in selected], [1, 4])

    def test_four_stream_tokenization_contract_is_receipt_bound(self) -> None:
        tokenization = self.experiment["data"]["tokenization"]
        self.assertEqual(tokenization["workers"], 64)
        self.assertTrue(tokenization["append_eod"])
        self.assertEqual(tokenization["streams"], {
            "hplt": "hplt_only_ext_text_document",
            "openarchives": "glossapi_only_ext_text_document",
            "foreign": "foreign_replay_only_ext_text_document",
            "old_greek": "old_greek_replay_only_ext_text_document",
        })
        wrapper = (ROOT / "clariden/tokenize_h2g_stream_debug.sbatch").read_text(encoding="utf-8")
        self.assertIn("#SBATCH --partition=debug", wrapper)
        self.assertIn('"${SLURM_JOB_PARTITION:-}" == debug', wrapper)
        self.assertIn("--append-eod --workers 64 --json-keys text", wrapper)
        self.assertIn("freeze_tokenized_stream.py", wrapper)
        self.assertIn("verify_data_runtime.py", wrapper)
        self.assertIn("quarantine_failure", wrapper)
        self.assertIn("H2G_MEGATRON_RECEIPT:?", wrapper)
        self.assertIn('megatron_receipt="$H2G_MEGATRON_RECEIPT"', wrapper)
        self.assertNotIn('receipts/training_megatron_runtime.json', wrapper)
        split = (ROOT / "scripts/split_replay_stage_b.py").read_text(encoding="utf-8")
        self.assertIn("06f244dd4e0d44f8352af14601768385d77fd35362b3bececda72c01de28f7aa", split)
        split_wrapper = (ROOT / "clariden/split_replay_stage_b_debug.sbatch").read_text(encoding="utf-8")
        self.assertIn("--post-greekmmlu-receipt", split_wrapper)
        self.assertIn("--post-native-corpus-receipt", split_wrapper)
        self.assertIn("--post-native-scan-receipt", split_wrapper)

    def test_historical_tokenizer_materialization_is_byte_only(self) -> None:
        materializer = (ROOT / "scripts/materialize_historical_tokenizer.py").read_text(encoding="utf-8")
        self.assertIn("shutil.copyfile(", materializer)
        self.assertNotIn("shutil.copy2(", materializer)
        self.assertIn("sha256_file(args.output_root / name)", materializer)

    def test_megatron_dataset_helper_is_compiled_bound_and_required(self) -> None:
        runtime_wrapper = (ROOT / "clariden/prepare_training_megatron_debug.sbatch").read_text(encoding="utf-8")
        finalizer = (ROOT / "scripts/finalize_training_megatron.py").read_text(encoding="utf-8")
        cache_wrapper = (ROOT / "clariden/build_phase_gptdataset_cache_debug.sbatch").read_text(encoding="utf-8")
        cache_builder = (ROOT / "scripts/build_phase_gptdataset_cache.py").read_text(encoding="utf-8")
        preflight = (ROOT / "scripts/preflight_train_segment.py").read_text(encoding="utf-8")
        self.assertIn('make -C "$output_root/megatron/core/datasets" -j 1', runtime_wrapper)
        self.assertIn("pytorch/v2.9.1:v2", runtime_wrapper)
        self.assertIn("helpers_cpp*.so", runtime_wrapper)
        self.assertIn('"dataset_helpers"', finalizer)
        self.assertIn("helper_import_smoke(root, helper)", finalizer)
        self.assertIn("python_cache_tag", finalizer)
        self.assertIn("H2G_MEGATRON_RECEIPT:?", cache_wrapper)
        self.assertIn("quarantine_failure", cache_wrapper)
        self.assertIn("--megatron-receipt", cache_wrapper)
        self.assertIn("require_helpers=True", cache_builder)
        self.assertIn("def single_rank_process_group()", cache_builder)
        self.assertIn('backend="gloo"', cache_builder)
        self.assertIn("world_size=1", cache_builder)
        self.assertIn("dist.destroy_process_group()", cache_builder)
        self.assertIn('"cache_build_process_group": process_group', cache_builder)
        self.assertIn('"cache_build_process_group": process_group', (ROOT / "scripts/freeze_phase_blend_cache.py").read_text(encoding="utf-8"))
        self.assertIn("phase cache single-rank process-group proof", (ROOT / "scripts/freeze_phase_blend_cache.py").read_text(encoding="utf-8"))
        self.assertIn("require_helpers=True", preflight)

    def test_phase3_tokenization_and_manifest_bind_exact_megatron_receipt(self) -> None:
        phase3 = (ROOT / "clariden/tokenize_phase3_stream_debug.sbatch").read_text(
            encoding="utf-8"
        )
        manifest = (
            ROOT / "clariden/freeze_pre_main_artifact_manifest_debug.sbatch"
        ).read_text(encoding="utf-8")
        for wrapper in (phase3, manifest):
            self.assertIn("H2G_MEGATRON_RECEIPT", wrapper)
            self.assertNotIn("receipts/training_megatron_runtime.json", wrapper)
        self.assertIn('megatron_receipt="$H2G_MEGATRON_RECEIPT"', phase3)
        self.assertIn(
            '"training_megatron_runtime=$H2G_MEGATRON_RECEIPT"', manifest
        )
        compatibility = (
            ROOT / "scripts/freeze_producer_bundle_compatibility.py"
        ).read_text(encoding="utf-8")
        self.assertIn(
            '"subprojects/08_targeted_8b_cpt_experiments/clariden/'
            'tokenize_h2g_stream_debug.sbatch"',
            compatibility,
        )
        self.assertIn(
            '"subprojects/08_targeted_8b_cpt_experiments/clariden/'
            'tokenize_phase3_stream_debug.sbatch"',
            compatibility,
        )

    def test_training_requires_allocation_free_static_preflight(self) -> None:
        preflight = (ROOT / "scripts/preflight_train_segment.py").read_text(encoding="utf-8")
        trainer = (ROOT / "clariden/train_hard_h_to_g_segment.sbatch").read_text(encoding="utf-8")
        self.assertIn('"--preallocation-static"', preflight)
        self.assertIn('"--static-preflight-receipt"', preflight)
        self.assertIn('"--preauthorization-manifest"', preflight)
        self.assertIn("static technical preflight must not self-authorize launch", preflight)
        self.assertIn("allocated preflight requires the final authorization gate", preflight)
        self.assertIn("allocated preflight requires the allocation-free static contract", preflight)
        self.assertGreaterEqual(preflight.count("verify_payload_hashes=False"), 3)
        self.assertGreaterEqual(preflight.count("accepted_code_bundles=accepted_code_bundles"), 3)
        self.assertIn("def code_bundle_identities(", preflight)
        self.assertIn("def validate_initialization_load_contract(", preflight)
        self.assertIn('"initialization reference model/config geometry drift"', preflight)
        self.assertIn('"initialization TP/PP geometry drift"', preflight)
        self.assertIn('"initialization RoPE scaling drift"', preflight)
        self.assertIn(
            'static.get("segment_contract") == segment_contract',
            preflight,
        )
        self.assertIn('"runtime_compat_contract": runtime_compat_contract', preflight)
        self.assertIn('"runtime_compat_receipt": file_binding(args.runtime_compat_receipt)', preflight)
        self.assertNotIn(
            '"runtime_compat_receipt": file_binding(args.runtime_compat_receipt),\n'
            '        "checkpoint_permit"',
            preflight,
        )
        self.assertIn("H2G_STATIC_PREFLIGHT:?", trainer)
        self.assertIn('--static-preflight-receipt "$H2G_STATIC_PREFLIGHT"', trainer)

    def test_megatron_index_inspection_reconciles_documents_and_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "fixture.idx"
            lengths = (4, 7)
            pointers = (0, 8)
            document_index = (0, 1, 2)
            payload = bytearray(INDEX_MAGIC)
            payload.extend(struct.pack("<Q", 1))
            payload.extend(struct.pack("<B", 3))
            payload.extend(struct.pack("<Q", len(lengths)))
            payload.extend(struct.pack("<Q", len(document_index)))
            payload.extend(struct.pack("<2i", *lengths))
            payload.extend(struct.pack("<2q", *pointers))
            payload.extend(struct.pack("<3q", *document_index))
            path.write_bytes(payload)
            observed = inspect_index(path)
            self.assertEqual(observed["documents"], 2)
            self.assertEqual(observed["sequences"], 2)
            self.assertEqual(observed["tokens_including_eod"], 11)

            input_jsonl = Path(temporary) / "input.jsonl"
            input_jsonl.write_text(
                json.dumps({"text": "alpha", "source_dataset": "source-a", "source_doc_id": "doc-a"}) + "\n"
                + json.dumps({"text": "beta", "source": "source-b", "doc_id": "doc-b"}) + "\n",
                encoding="utf-8",
            )
            catalog = Path(temporary) / "catalog.jsonl"
            binding = write_document_catalog(input_jsonl, path, "hplt", catalog)
            self.assertEqual((binding["rows"], binding["tokens_including_eod"]), (2, 11))
            rows = [json.loads(line) for line in catalog.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([row["document_index"] for row in rows], [0, 1])
            self.assertEqual([row["token_count"] for row in rows], [4, 7])

    def test_realized_ledger_follows_shuffle_sample_document_indices(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            description = root / "fixture-description.txt"
            description.write_text("{}", encoding="utf-8")
            # Sample 0 spans document-index positions 0..1; sample 1 is pos 1;
            # sample 2 is pos 2. Shuffle maps component samples [0,1] -> [2,0].
            import numpy as np
            np.save(root / "fixture-document_index.npy", np.array([4, 8, 12], dtype=np.int32), allow_pickle=False)
            np.save(root / "fixture-sample_index.npy", np.array([[0, 0], [1, 0], [2, 0], [2, 9]], dtype=np.int64), allow_pickle=False)
            np.save(root / "fixture-shuffle_index.npy", np.array([2, 0, 1], dtype=np.int64), allow_pickle=False)
            self.assertEqual(documents_for_component(np.array([0, 1], dtype=np.int64), description), {4, 8, 12})

    def test_1p5b_td_reuses_historical_batch_and_seed(self) -> None:
        td = self.experiment["initialization"]["token_distillation"]
        self.assertEqual(td["batch_size"], 8)
        self.assertEqual(td["seed"], 20260523)
        wrapper = (ROOT / "clariden/run_1p5b_td_init_common.sh").read_text(encoding="utf-8")
        self.assertIn("--batch-size 8", wrapper)
        self.assertIn("--seed 20260523", wrapper)
        self.assertIn("td_work=$(mktemp -d", wrapper)
        snippets = (ROOT / "clariden/build_td_snippets_debug.sbatch").read_text(encoding="utf-8")
        self.assertIn("--coverage-summary", snippets)
        self.assertIn("--coverage-seed 20260523", snippets)
        self.assertIn("td_coverage_prepass_batched.py", snippets)
        self.assertIn("--parity-documents 256", snippets)
        self.assertIn("--encode-workers 16", snippets)
        self.assertIn("--max-encode-batches-in-flight 32", snippets)
        freezer = (ROOT / "scripts/freeze_td_training_inputs.py").read_text(encoding="utf-8")
        self.assertIn("coverage batched-encoding parity authority drift", freezer)
        self.assertIn("TD coverage input order or identity drift", freezer)
        self.assertIn("--dataset-authority", snippets)
        self.assertIn("numpy_flatnonzero_contiguous_id_range_preserving_ascending_order", freezer)
        self.assertIn("deferred_until_exact_pinned_reservoir_slot_is_selected", freezer)

    def test_initialization_lineage_is_receipt_bound_end_to_end(self) -> None:
        retok = (ROOT / "scripts/build_retok_reference_init.py").read_text(encoding="utf-8")
        td = (ROOT / "scripts/verify_td_initialization.py").read_text(encoding="utf-8")
        geometry = (ROOT / "scripts/prepare_training_geometry_hf.py").read_text(encoding="utf-8")
        roundtrip = (ROOT / "scripts/finalize_init_roundtrip.py").read_text(encoding="utf-8")
        td_wrapper = "\n".join((
            (ROOT / "clariden/build_1p5b_td_init_debug.sbatch").read_text(encoding="utf-8"),
            (ROOT / "clariden/run_1p5b_td_init_common.sh").read_text(encoding="utf-8"),
        ))
        geometry_wrapper = (ROOT / "clariden/prepare_init_geometry_debug.sbatch").read_text(encoding="utf-8")
        roundtrip_wrapper = (ROOT / "clariden/roundtrip_td_init_debug.sbatch").read_text(encoding="utf-8")
        self.assertIn("--base-materialization-receipt", retok)
        self.assertIn("--tokenizer-compatibility", retok)
        self.assertIn("base-vocabulary content-difference contract drift", retok)
        self.assertIn("base_rows_preserved_by_id_without_permutation", retok)
        self.assertIn("require_relative_inventory", retok)
        self.assertIn("shutil.copyfile", retok)
        self.assertNotIn("shutil.copy2", retok)
        self.assertIn("--parent-materialization-receipt", td)
        self.assertIn("--td-training-inputs-receipt", td)
        self.assertIn("--acceptance-policy", td)
        self.assertIn("--policy-authorization", td)
        self.assertIn("--reference-objective-probe", td)
        self.assertIn("architecture-local absolute row-norm safety", td)
        self.assertIn('"td_model_files": td_model_files', td)
        self.assertIn("--source-authority", geometry)
        self.assertIn('"source_authority": file_binding', geometry)
        self.assertIn('"rope_type": "llama3"', geometry)
        self.assertIn('"factor": 8.0', geometry)
        self.assertIn("--geometry-receipt", roundtrip)
        self.assertIn('"geometry_receipt": file_binding', roundtrip)
        self.assertIn("--parent-materialization-receipt", td_wrapper)
        self.assertIn("configs/1p5b_tokenizer_compatibility_v1.json", td_wrapper)
        self.assertIn("--source-authority", geometry_wrapper)
        self.assertIn("--geometry-receipt", roundtrip_wrapper)

    def test_retok_loader_resolves_only_its_frozen_sibling_common_module(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "_common.py").write_text("VALUE = 41\n", encoding="utf-8")
            tool = root / "retok.py"
            tool.write_text(
                "from _common import VALUE\n"
                "def compute_retok_init(**kwargs):\n"
                "    return VALUE + kwargs['increment']\n",
                encoding="utf-8",
            )
            prior_path = list(sys.path)
            prior_common = sys.modules.get("_common")
            loaded = load_retok(tool)
            self.assertEqual(loaded(increment=1), 42)
            self.assertEqual(sys.path, prior_path)
            self.assertIs(sys.modules.get("_common"), prior_common)

    def test_frozen_retok_bundle_includes_and_loads_canonical_common_sibling(self) -> None:
        deploy = (ROOT / "clariden/deploy_targeted_bundle.sh").read_text(encoding="utf-8")
        self.assertIn("retok_common_source=$init_bakeoff_root/arms/_common.py", deploy)
        self.assertIn('frozen_td_tools/_common.py', deploy)
        frozen_root = ROOT.parents[1] / "frozen_td_tools"
        if frozen_root.is_dir():
            self.assertTrue((frozen_root / "_common.py").is_file())
            self.assertTrue(callable(load_retok(frozen_root / "retok.py")))

    def test_artifact_inventory_verifier_is_exact_and_detects_byte_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "config.json"
            second = root / "nested" / "weights.bin"
            second.parent.mkdir()
            first.write_text("{}\n", encoding="utf-8")
            second.write_bytes(b"weights")
            rows = [
                {"path": str(path.relative_to(root)), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
                for path in (first, second)
            ]
            self.assertEqual(len(require_relative_inventory(root=root, rows=rows)), 2)
            second.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "size drift|SHA-256 drift"):
                require_relative_inventory(root=root, rows=rows)

    def test_batched_td_coverage_is_byte_exact_to_sequential_reference(self) -> None:
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace

        frozen_reference = ROOT.parents[1] / "frozen_td_tools/td_coverage_prepass.py"
        local_reference = (
            ROOT.parent
            / "03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/"
            "init_bakeoff/token_distillation/td_coverage_prepass.py"
        )
        reference = frozen_reference if frozen_reference.is_file() else local_reference
        self.assertTrue(reference.is_file())
        batched = ROOT / "scripts/td_coverage_prepass_batched.py"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base_dir = root / "base"
            student_dir = root / "student"
            base_dir.mkdir()
            student_dir.mkdir()

            def write_tokenizer(path: Path, vocab: dict[str, int]) -> None:
                tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
                tokenizer.pre_tokenizer = Whitespace()
                tokenizer.save(str(path / "tokenizer.json"))

            write_tokenizer(base_dir, {"[UNK]": 0, "a": 1, "b": 2, "c": 3})
            write_tokenizer(student_dir, {"[UNK]": 0, "a": 1, "b": 2, "c": 3, "ab": 4, "bc": 5})
            corpus = root / "corpus.jsonl"
            corpus.write_text("".join(
                json.dumps({"text": text, "source": "fixture", "doc_id": f"d{i}", "lang": "el"}) + "\n"
                for i, text in enumerate(("ab a bc", "c ab b", "bc ab a", "ab bc c", "a ab bc"))
            ), encoding="utf-8")
            sequential_dir = root / "sequential"
            batched_dir = root / "batched"
            common = [
                "--input-jsonl", str(corpus), "--base-tokenizer", str(base_dir),
                "--student-tokenizer", str(student_dir), "--new-id-start", "4",
                "--new-id-end", "6", "--target-extended-tokens", "12",
                "--snippets-per-token", "3", "--snippet-token-radius", "2",
                "--progress-token-interval", "0", "--seed", "20260523",
            ]
            subprocess.run([sys.executable, str(reference), *common, "--output-dir", str(sequential_dir)], check=True, capture_output=True, text=True)
            subprocess.run([
                sys.executable, str(batched), "--reference-script", str(reference), *common,
                "--output-dir", str(batched_dir), "--encode-batch-documents", "3",
                "--encode-batch-characters", "100", "--encode-workers", "2",
                "--max-encode-batches-in-flight", "4",
                "--parity-documents", "5",
            ], check=True, capture_output=True, text=True)
            for relative in ("td_coverage_prepass.jsonl", "td_snippet_index/snippets.jsonl"):
                self.assertEqual((sequential_dir / relative).read_bytes(), (batched_dir / relative).read_bytes())
            sequential_summary = read_json(sequential_dir / "td_coverage_summary.json")
            batched_summary = read_json(batched_dir / "td_coverage_summary.json")
            for key in (
                "tokens_scanned", "docs_seen", "docs_used", "chars_seen", "stopped_on_budget",
                "non_nfc_docs", "status_counts", "action_counts", "enough_100_fraction",
                "enough_25_fraction", "low_lt25_count", "recommended_next_step",
            ):
                self.assertEqual(sequential_summary[key], batched_summary[key], key)
            self.assertEqual(
                batched_summary["encoding_execution"]["scientific_state_update_order"],
                "identical_to_pinned_sequential_reference",
            )

    def test_receipt_bound_replay_glob_must_expand_exactly(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selected = [root / "000_00122.parquet", root / "000_00236.parquet"]
            for path in selected:
                path.write_bytes(b"parquet-placeholder")
            self.assertEqual(receipt_bound_glob(selected), str(root.resolve() / "*.parquet"))
            (root / "unreceipted.parquet").write_bytes(b"extra")
            with self.assertRaisesRegex(ValueError, "not exact"):
                receipt_bound_glob(selected)

    def test_validation_panel_identity_adapters_are_explicit_and_fail_closed(self) -> None:
        standard = extract_panel_row("hplt", {
            "text": "alpha",
            "doc_id": "hplt-doc",
            "source_dataset": "hplt",
        }, 1)
        self.assertEqual(standard, {
            "text": "alpha",
            "doc_id": "hplt-doc",
            "cluster_id": "hplt-doc",
            "source_dataset": "hplt",
        })
        neutral = extract_panel_row("neutral_external_modern_greek", {
            "text": "beta",
            "source_doc_id": "gpp-doc",
            "cluster_id": "gpp-cluster",
            "source_id": "zenodo_greek_parliament_proceedings_2587904",
        }, 1)
        self.assertEqual(neutral["doc_id"], "gpp-doc")
        self.assertEqual(neutral["cluster_id"], "gpp-cluster")
        self.assertEqual(neutral["source_dataset"], "zenodo_greek_parliament_proceedings_2587904")
        with self.assertRaisesRegex(ValueError, "missing/empty source_doc_id"):
            extract_panel_row("neutral_external_modern_greek", {
                "text": "beta",
                "doc_id": "ambiguous-fallback-is-forbidden",
                "cluster_id": "gpp-cluster",
                "source_id": "gpp",
            }, 1)

    def test_phase_local_cursor_drift_fails(self) -> None:
        value = copy.deepcopy(self.experiment)
        value["schedule"]["extension"]["phase_local_cursor_at_3218"] += 1024
        with self.assertRaisesRegex(ValueError, "3218 phase cursor drift"):
            validate_experiment(value)

    def test_runtime_phase_cursor_contract_covers_phase2_and_phase3(self) -> None:
        base = {
            "PHASE_LOCAL_DATA_INDEX": "1",
            "GLOBAL_BATCH_SIZE": "1024",
            "EXPECTED_DATA_CACHE_SHA256": "a" * 64,
            "ACTUAL_DATA_CACHE_SHA256": "a" * 64,
        }
        with mock.patch.dict("os.environ", {
            **base,
            "PHASE_START_UPDATE": "2261",
            "EXPECTED_GLOBAL_UPDATE": "2380",
            "EXPECTED_PHASE_LOCAL_CONSUMED_SAMPLES": str(119 * 1024),
        }, clear=True):
            self.assertEqual(validate_phase_cursor(), (2380 * 1024, 119 * 1024))
        with mock.patch.dict("os.environ", {
            **base,
            "PHASE_START_UPDATE": "3218",
            "EXPECTED_GLOBAL_UPDATE": "3456",
            "EXPECTED_PHASE_LOCAL_CONSUMED_SAMPLES": str(238 * 1024),
        }, clear=True):
            self.assertEqual(validate_phase_cursor(), (3456 * 1024, 238 * 1024))
        with mock.patch.dict("os.environ", {
            **base,
            "PHASE_START_UPDATE": "3218",
            "EXPECTED_GLOBAL_UPDATE": "3456",
            "EXPECTED_PHASE_LOCAL_CONSUMED_SAMPLES": "0",
        }, clear=True):
            with self.assertRaisesRegex(RuntimeError, "phase-local cursor drift"):
                validate_phase_cursor()

    def test_phase3_dataset_horizon_is_decoupled_from_scheduler_horizon(self) -> None:
        adjusted, changed = override_train_dataset_samples([3_782_656, 0, 0], 3_782_656, 487_424)
        self.assertTrue(changed)
        self.assertEqual(adjusted, [487_424, 0, 0])
        untouched, changed = override_train_dataset_samples([3_295_898, 0, 0], 3_782_656, 487_424)
        self.assertFalse(changed)
        self.assertEqual(untouched, [3_295_898, 0, 0])
        extension = self.experiment["schedule"]["extension"]
        self.assertEqual(extension["component_requested_samples_with_1p005_margin"], COMPONENT_REQUESTED_SAMPLES)
        self.assertEqual(PHASE3_COMPONENT_REQUESTED_SAMPLES, {
            "active_modern": COMPONENT_REQUESTED_SAMPLES["openarchives"],
            "foreign_replay": COMPONENT_REQUESTED_SAMPLES["foreign_replay"],
            "old_greek_replay": COMPONENT_REQUESTED_SAMPLES["old_greek_replay"],
        })
        self.assertEqual({
            pool: samples * SEQUENCE_LENGTH + 1
            for pool, samples in COMPONENT_REQUESTED_SAMPLES.items()
        }, {
            "openarchives": 1_585_115_137,
            "foreign_replay": 401_297_409,
            "old_greek_replay": 20_066_305,
        })

    def test_phase_data_path_is_ordered_weighted_and_file_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            roles = (
                ("active_modern", "1.0"),
                ("foreign_replay", "0.253164557"),
                ("old_greek_replay", "0.012658228"),
            )
            streams = ("phase3_openarchives", "phase3_foreign", "phase3_old_greek")
            components = []
            tokens = []
            for index, (role, weight) in enumerate(roles):
                prefix = (root / f"part{index}").resolve()
                Path(f"{prefix}.bin").write_bytes(bytes([index]))
                Path(f"{prefix}.idx").write_bytes(bytes([index + 1]))
                files = [file_binding(Path(f"{prefix}.bin")), file_binding(Path(f"{prefix}.idx"))]
                receipt = root / f"tokenized-{index}.json"
                receipt.write_text(json.dumps({
                    "schema_version": "apertus_hard_h_to_g_tokenized_stream_v1",
                    "status": "frozen",
                    "stream": streams[index],
                    "dataset_prefix": str(prefix),
                    "files": {"bin": files[0], "idx": files[1]},
                }), encoding="utf-8")
                components.append({
                    "role": role,
                    "weight": weight,
                    "prefix": str(prefix),
                    "files": files,
                    "tokenized_receipt": file_binding(receipt),
                })
                tokens.extend((weight, str(prefix)))
            spec = {
                "schema_version": "apertus_hard_h_to_g_phase_data_path_v1",
                "status": "frozen",
                "phase": 3,
                "components": components,
                "data_path_tokens": tokens,
                "data_path_shell_string": " ".join(tokens),
            }
            observed, prefixes = validate_data_path_spec(spec, 3)
            self.assertEqual(observed, tokens)
            self.assertEqual(len(prefixes), 3)
            spec["components"][1]["weight"] = "0.25"
            with self.assertRaisesRegex(ValueError, "weight drift"):
                validate_data_path_spec(spec, 3)

    def test_phase3_catalog_excludes_phase2_and_rejects_repeated_text(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ledger = root / "phase2.jsonl"
            ledger.write_text(json.dumps({
                "document_key_sha256": "1" * 64,
                "document_text_sha256": "2" * 64,
            }) + "\n", encoding="utf-8")
            candidate = root / "oa.jsonl"
            candidate.write_text("".join(json.dumps(row) + "\n" for row in (
                {"pool": "openarchives", "document_key_sha256": "1" * 64, "document_text_sha256": "3" * 64, "token_count": 10},
                {"pool": "openarchives", "document_key_sha256": "4" * 64, "document_text_sha256": "5" * 64, "token_count": 20},
                {"pool": "openarchives", "document_key_sha256": "6" * 64, "document_text_sha256": "5" * 64, "token_count": 30},
            )), encoding="utf-8")
            keys, texts, _ = phase2_identities(ledger)
            counters = __import__("collections").Counter()
            rows = eligible_rows(candidate, "openarchives", keys, texts, set(), set(), counters)
            self.assertEqual(next(rows)["document_key_sha256"], "4" * 64)
            with self.assertRaisesRegex(ValueError, "repeated Phase-3 document text"):
                next(rows)
            self.assertEqual(counters["phase2_overlap_rows"], 1)

    def test_allocation_disables_lr_pilot_and_keeps_fixed_matched_recipe(self) -> None:
        result = validate_allocation(copy.deepcopy(self.allocation))
        self.assertEqual(result["8b_world_size"], 64)
        self.assertEqual(result["1p5b_candidate_world_sizes"], {
            "1p5b_tp1_1node": 4,
            "1p5b_tp1_2node": 8,
            "1p5b_tp1_4node": 16,
        })
        self.assertEqual(self.allocation["learning_rate_policy"]["allocations"], 0)
        self.assertEqual(self.allocation["learning_rate_policy"]["peak_lr"], "5.5e-5")
        value = copy.deepcopy(self.allocation)
        value["learning_rate_policy"]["allocations"] = 1
        with self.assertRaisesRegex(ValueError, "allocation LR policy drift"):
            validate_allocation(value)

    def test_hamilton_closes_and_respects_capacity(self) -> None:
        result = hamilton({"a": 5, "b": 3, "c": 2}, 7)
        self.assertEqual(sum(result.values()), 7)
        self.assertTrue(all(result[key] <= cap for key, cap in {"a": 5, "b": 3, "c": 2}.items()))

    def test_hamilton_ties_do_not_depend_on_mapping_order(self) -> None:
        left = hamilton({"a": 1, "b": 1, "c": 1}, 2)
        right = hamilton({"c": 1, "b": 1, "a": 1}, 2)
        self.assertEqual(left, right)

    def test_sentinel_is_deterministic_nested_and_subject_complete(self) -> None:
        rows = []
        for subject in ("A", "B", "C"):
            for index in range(10):
                rows.append({
                    "example_id": f"{subject}:{index}",
                    "subject": subject,
                    "educational_level": "L1" if index % 2 == 0 else "L2",
                })
        left, use_level = select_nested(copy.deepcopy(rows), [9, 18])
        right, use_level_again = select_nested(list(reversed(copy.deepcopy(rows))), [9, 18])
        self.assertTrue(use_level and use_level_again)
        self.assertEqual(
            [row["example_id"] for row in left[9]],
            [row["example_id"] for row in right[9]],
        )
        ids9 = {row["example_id"] for row in left[9]}
        ids18 = {row["example_id"] for row in left[18]}
        self.assertLess(ids9, ids18)
        self.assertEqual({row["subject"] for row in left[9]}, {"A", "B", "C"})

    def test_sentinel_falls_back_to_subject_if_any_level_is_missing(self) -> None:
        rows = [
            {"example_id": "a", "subject": "A", "educational_level": "L"},
            {"example_id": "b", "subject": "B", "educational_level": None},
            {"example_id": "c", "subject": "A", "educational_level": "L"},
            {"example_id": "d", "subject": "B", "educational_level": "L"},
        ]
        selected, use_level = select_nested(rows, [2, 3])
        self.assertFalse(use_level)
        self.assertEqual({row["subject"] for row in selected[2]}, {"A", "B"})

    def test_three_sentinel_sizes_are_monotonic_under_reapportionment_pressure(self) -> None:
        rows = []
        for subject, count in (("A", 17), ("B", 11), ("C", 7), ("D", 5)):
            for index in range(count):
                rows.append({
                    "example_id": f"{subject}:{index}",
                    "subject": subject,
                    "educational_level": "L1" if index % 3 else "L2",
                })
        selected, _ = select_nested(rows, [9, 17, 31])
        ids = [{row["example_id"] for row in selected[size]} for size in (9, 17, 31)]
        self.assertLess(ids[0], ids[1])
        self.assertLess(ids[1], ids[2])

    def test_hash_uses_nul_separator(self) -> None:
        import hashlib

        expected = hashlib.sha256(b"greekmmlu-sentinel-v1\0greekmmlu:7").hexdigest()
        self.assertEqual(stable_digest("greekmmlu:7"), expected)

    def test_choice_nll_uses_length_normalized_scores(self) -> None:
        row = {
            "answer_index": 1,
            "choice_scores": [
                {"avg_logprob": -2.0, "sum_logprob": -20.0, "num_tokens": 10},
                {"avg_logprob": -1.0, "sum_logprob": -2.0, "num_tokens": 2},
            ],
        }
        expected = math.log(math.exp(-2.0) + math.exp(-1.0)) - (-1.0)
        self.assertAlmostEqual(choice_nll(row), expected)

    def test_asset_inventory_is_fail_closed_and_hash_bound(self) -> None:
        import hashlib

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            asset = root / "asset.bin"
            asset.write_bytes(b"frozen")
            expected = hashlib.sha256(b"frozen").hexdigest()
            row = inspect_asset("asset", {
                "path": str(asset),
                "kind": "file",
                "required": True,
                "hash": True,
                "expected_sha256": expected,
            })
            self.assertEqual(row["sha256"], expected)
            with self.assertRaisesRegex(ValueError, "SHA-256 drift"):
                inspect_asset("asset", {
                    "path": str(asset),
                    "kind": "file",
                    "required": True,
                    "hash": True,
                    "expected_sha256": "0" * 64,
                })
            missing = inspect_asset("purged", {
                "path": str(root / "purged.bin"),
                "kind": "file",
                "required": False,
                "expected_missing": True,
            })
            self.assertTrue(missing["expected_missing"])

    def test_lr_receipt_arithmetic_uses_exact_decimals(self) -> None:
        self.assertTrue(decimal_product_equal("5.5e-6", "5.5e-5", "0.1"))
        self.assertFalse(decimal_product_equal("5.4e-6", "5.5e-5", "0.1"))

    def test_new_debug_wrappers_use_proven_python_for_bundle_verification(self) -> None:
        for name in (
            "inventory_hard_h_to_g_assets_debug.sbatch",
            "inspect_native_overlap_schema_debug.sbatch",
        ):
            text = (ROOT / "clariden" / name).read_text(encoding="utf-8")
            self.assertIn("#SBATCH --partition=debug", text)
            self.assertIn("/usr/bin/python3.11", text)
            self.assertNotIn('\npython3 "$H2G_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py"', text)

    def test_r2_operational_builders_are_forced_to_debug(self) -> None:
        for name in (
            "build_replay_scan_input_debug.sbatch",
            "inspect_hf_release_debug.sbatch",
            "run_fresh_greekmmlu_stream_scan_debug.sbatch",
            "filter_replay_greekmmlu_debug.sbatch",
            "audit_replay_stage_b_greekmmlu_debug.sbatch",
            "materialize_phase3_openarchives_candidates_debug.sbatch",
            "prepare_phase3_openarchives_candidates_debug.sbatch",
            "anonymize_phase3_openarchives_candidates_debug.sbatch",
            "build_phase3_openarchives_catalog_debug.sbatch",
            "build_phase3_unseen_catalog_debug.sbatch",
            "inspect_replay_parquet_debug.sbatch",
            "build_hard_h_to_g_source_views_debug.sbatch",
            "freeze_replay_scan_adapter_debug.sbatch",
            "freeze_replay_source_inventory_debug.sbatch",
            "freeze_reused_validation_panels_debug.sbatch",
            "filter_replay_validation_exclusions_debug.sbatch",
            "build_replay_selection_debug.sbatch",
            "freeze_native_suite_scan_authority_debug.sbatch",
            "run_native_suite_replay_scan_debug.sbatch",
            "filter_replay_native_exclusions_debug.sbatch",
            "materialize_replay_postfilter_scan_debug.sbatch",
            "prepare_greek_source_stream_debug.sbatch",
            "split_replay_stage_b_debug.sbatch",
            "tokenize_h2g_stream_debug.sbatch",
            "tokenize_phase3_stream_debug.sbatch",
            "export_realized_document_ledger_debug.sbatch",
            "anonymize_training_stream_debug.sbatch",
            "materialize_init_model_debug.sbatch",
            "freeze_td_norm_contract_debug.sbatch",
            "build_td_snippets_debug.sbatch",
            "build_1p5b_td_init_debug.sbatch",
            "prepare_init_geometry_debug.sbatch",
            "roundtrip_td_init_debug.sbatch",
            "prepare_training_megatron_debug.sbatch",
            "freeze_phase_blend_cache_debug.sbatch",
            "freeze_online_validation_binaries_debug.sbatch",
            "audit_training_checkpoint_debug.sbatch",
            "build_phase_data_path_spec_debug.sbatch",
            "build_phase_gptdataset_cache_debug.sbatch",
            "materialize_phase_cache_debug.sbatch",
            "freeze_greekmmlu_sentinels_debug.sbatch",
            "validate_greekmmlu_sentinels_debug.sbatch",
            "freeze_cross_scale_sentinel_authority_debug.sbatch",
            "freeze_pre_main_data_authorities_debug.sbatch",
            "freeze_cross_scale_realized_ledger_debug.sbatch",
            "freeze_phase3_authority_debug.sbatch",
            "freeze_producer_bundle_compatibility_debug.sbatch",
            "freeze_prelaunch_benchmark_contract_debug.sbatch",
            "finalize_profile_benchmark_debug.sbatch",
            "finalize_lr_pilot_arm_debug.sbatch",
            "freeze_profile_promotion_debug.sbatch",
            "freeze_lr_selection_debug.sbatch",
            "freeze_training_run_permit_debug.sbatch",
            "freeze_production_timing_and_allocation_debug.sbatch",
            "finalize_phase3_resume_smoke_debug.sbatch",
            "finalize_matched_study_evidence_debug.sbatch",
            "freeze_post_checkpoint_authorities_debug.sbatch",
        ):
            text = (ROOT / "clariden" / name).read_text(encoding="utf-8")
            self.assertIn("#SBATCH --partition=debug", text)
            self.assertIn('"${SLURM_JOB_PARTITION:-}" == debug', text)
            self.assertIn("verify_code_bundle.py", text)

    def test_prelaunch_runner_is_normal_and_resume_uses_uninterrupted_update_one(self) -> None:
        text = (ROOT / "clariden/run_prelaunch_benchmark.sbatch").read_text(encoding="utf-8")
        self.assertIn("#SBATCH --partition=normal", text)
        self.assertIn('[[ "${SLURM_JOB_PARTITION:-}" == normal ]]', text)
        self.assertIn("freeze_update_one_load_view", text)
        self.assertIn('restart/uninterrupted/checkpoints', text)
        self.assertIn('restart/load_update_1', text)
        self.assertIn('restart/phase2_uninterrupted', text)
        self.assertIn('restart/phase2_resumed', text)
        self.assertIn('phase_local_samples=1024', (ROOT / "scripts/finalize_profile_benchmark.py").read_text(encoding="utf-8"))
        self.assertNotIn('restart/source', text)
        self.assertIn('H2G_PHASE_CACHE_ROOT="$cache_root"', text)

    def test_td_normal_reroute_preserves_the_exact_scientific_body(self) -> None:
        debug = (ROOT / "clariden/build_1p5b_td_init_debug.sbatch").read_text(encoding="utf-8")
        normal = (ROOT / "clariden/build_1p5b_td_init_normal.sbatch").read_text(encoding="utf-8")
        common = (ROOT / "clariden/run_1p5b_td_init_common.sh").read_text(encoding="utf-8")
        self.assertIn("#SBATCH --partition=debug", debug)
        self.assertIn('"${SLURM_JOB_PARTITION:-}" == debug', debug)
        self.assertIn("#SBATCH --partition=normal", normal)
        self.assertIn('"${SLURM_JOB_PARTITION:-}" == normal', normal)
        self.assertIn("#SBATCH --time=04:00:00", normal)
        self.assertIn("cannot fit the 01:30 debug limit", debug)
        self.assertNotIn("run_1p5b_td_init_common.sh", debug)
        for wrapper in (debug, normal):
            self.assertIn("verify_code_bundle.py", wrapper)
            self.assertNotIn("train_retok_td.py", wrapper)
        self.assertIn("run_1p5b_td_init_common.sh", normal)
        self.assertIn("--epochs 1 --batch-size 8", common)
        self.assertIn("--learning-rate 1e-4 --target-layer 6", common)
        self.assertIn("--dtype bfloat16 --device cuda --seed 20260523", common)
        self.assertIn("1p5b_td_policy_authorization.json", common)
        self.assertIn("evaluate_td_objective.py", common)
        self.assertIn("1p5b_td_hf_raw_v2", common)
        compatibility = (ROOT / "scripts/freeze_producer_bundle_compatibility.py").read_text(encoding="utf-8")
        self.assertIn('"subprojects/08_targeted_8b_cpt_experiments/clariden/build_1p5b_td_init_normal.sbatch"', compatibility)
        self.assertIn('"subprojects/08_targeted_8b_cpt_experiments/clariden/run_1p5b_td_init_common.sh"', compatibility)

    def test_bundle_carries_the_historical_8b_lr_decision_consumed_by_the_freezer(self) -> None:
        deploy = (ROOT / "clariden/deploy_targeted_bundle.sh").read_text(encoding="utf-8")
        freezer = (ROOT / "clariden/freeze_lr_selection_debug.sbatch").read_text(encoding="utf-8")
        expected = "subprojects/05_token_distillation_cpt/PRODUCTION_LR_DECISION_20260613.md"
        self.assertIn(expected, deploy)
        self.assertIn(expected, freezer)
        self.assertIn("historical_lr_decision_source", deploy)
        compatibility = (ROOT / "scripts/freeze_producer_bundle_compatibility.py").read_text(
            encoding="utf-8"
        )
        self.assertIn(expected, compatibility)

    def test_bundle_excludes_local_runtime_and_test_caches(self) -> None:
        deploy = (ROOT / "clariden/deploy_targeted_bundle.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("--delete-excluded", deploy)
        for cache in ("__pycache__", ".ruff_cache", ".pytest_cache"):
            self.assertIn(cache, deploy)

    def test_profile_reference_uses_the_frozen_one_node_id(self) -> None:
        text = (ROOT / "scripts/finalize_profile_benchmark.py").read_text(encoding="utf-8")
        self.assertIn('1p5b_tp1_1node', text)
        self.assertNotIn('1p5b_1node_dp4', text)

    def test_production_cannot_inherit_the_phase2_smoke_override(self) -> None:
        text = (ROOT / "clariden/train_hard_h_to_g_segment.sbatch").read_text(encoding="utf-8")
        self.assertIn("unset H2G_PHASE_START_UPDATE_OVERRIDE H2G_PRELAUNCH_PHASE2_RESTART_SMOKE", text)

    def test_phase3_resume_smoke_uses_normal_but_finalization_uses_debug(self) -> None:
        runner = (ROOT / "clariden/run_phase3_resume_smoke.sbatch").read_text(encoding="utf-8")
        finalizer = (ROOT / "clariden/finalize_phase3_resume_smoke_debug.sbatch").read_text(encoding="utf-8")
        self.assertIn("#SBATCH --partition=normal", runner)
        self.assertIn("--one-update-resume-smoke", runner)
        self.assertIn("#SBATCH --partition=debug", finalizer)
        self.assertIn("verify_data_runtime.py", finalizer)

    def test_production_timing_requires_measured_wall_and_test_only_evidence(self) -> None:
        finalizer = (ROOT / "scripts/freeze_production_timing_and_allocation.py").read_text(encoding="utf-8")
        wrapper = (ROOT / "clariden/freeze_production_timing_and_allocation_debug.sbatch").read_text(encoding="utf-8")
        self.assertIn("production_cadence_wall_seconds", finalizer)
        self.assertIn("CONSERVATIVE_MULTIPLIER = 1.15", finalizer)
        self.assertIn("--test-only", wrapper)
        self.assertIn("maximum_pending_delayed_successors", finalizer)
        self.assertIn('parser.add_argument("--scale"', finalizer)

    def test_runtime_authority_wrappers_require_explicit_immutable_paths(self) -> None:
        expected = {
            "freeze_lr_selection_debug.sbatch": ("H2G_LR_SELECTION_OUTPUT",),
            "freeze_profile_promotion_debug.sbatch": (
                "H2G_PROFILE_PROMOTION_OUTPUT",
            ),
            "freeze_training_run_permit_debug.sbatch": (
                "H2G_PROFILE_PROMOTION",
                "H2G_LR_SELECTION",
                "H2G_TRAINING_RUN_PERMIT_OUTPUT",
            ),
            "freeze_production_timing_and_allocation_debug.sbatch": (
                "H2G_TRAINING_RUN_PERMIT",
                "H2G_PROFILE_PROMOTION",
                "H2G_SUBMISSION_DRY_RUN_OUTPUT",
                "H2G_PRODUCTION_TIMING_OUTPUT",
                "H2G_ALLOCATION_SCHEDULE_OUTPUT",
                "H2G_SUBMISSION_DRY_RUN_LOG",
            ),
            "build_canonical_campaign_contracts_debug.sbatch": (
                "H2G_TRAINING_RUN_PERMIT",
                "H2G_PROFILE_PROMOTION",
                "H2G_PRODUCTION_TIMING",
                "H2G_ALLOCATION_SCHEDULE",
                "H2G_PRE_MAIN_LAUNCH_GATE",
            ),
            "finalize_profile_benchmark_debug.sbatch": (
                "H2G_PROFILE_BENCHMARK_OUTPUT",
            ),
            "finalize_lr_pilot_arm_debug.sbatch": ("H2G_LR_PILOT_ARM_OUTPUT",),
        }
        for name, variables in expected.items():
            text = (ROOT / "clariden" / name).read_text(encoding="utf-8")
            self.assertIn('"$H2G_STAGE_ROOT"/receipts/*', text)
            for variable in variables:
                self.assertIn(variable, text)

        compiler = (ROOT / "scripts/build_canonical_campaign_contracts.py").read_text(
            encoding="utf-8"
        )
        for option in (
            "--training-run-permit",
            "--profile-promotion",
            "--production-timing",
            "--allocation-schedule",
            "--pre-main-launch-gate",
        ):
            self.assertIn(f'parser.add_argument("{option}", type=Path, required=True)', compiler)

        compatibility = (
            ROOT / "scripts/freeze_producer_bundle_compatibility.py"
        ).read_text(encoding="utf-8")
        for name in (
            "finalize_lr_pilot_arm_debug.sbatch",
            "finalize_profile_benchmark_debug.sbatch",
            "freeze_lr_selection_debug.sbatch",
            "freeze_profile_promotion_debug.sbatch",
            "freeze_training_run_permit_debug.sbatch",
        ):
            self.assertIn(name, compatibility)

    def test_lr_pilot_validation_parser_requires_exact_baseline_and_endpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / "driver.out"
            rows = []
            for iteration in (0, 238):
                for panel in ("hplt", "english", "de", "ru", "zh", "code", "old_greek"):
                    rows.append(
                        f"0: validation loss at iteration {iteration} [{panel}] | "
                        f"lm loss value: {1.5 - iteration / 1000:.6f} |\n"
                    )
            log.write_text("".join(rows), encoding="utf-8")
            parsed = parse_validation_log(log)
            self.assertEqual(len(parsed), 14)
            log.write_text("".join(rows[:-1]), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "coverage drift"):
                parse_validation_log(log)

    def test_r2_data_wrappers_require_the_verified_runtime(self) -> None:
        wrappers = (
            "anonymize_training_stream_debug.sbatch",
            "build_hard_h_to_g_source_views_debug.sbatch",
            "build_modern_mix_selection_debug.sbatch",
            "build_native_suite_training_exclusions_debug.sbatch",
            "build_replay_scan_input_debug.sbatch",
            "build_replay_selection_debug.sbatch",
            "filter_replay_native_exclusions_debug.sbatch",
            "filter_replay_validation_exclusions_debug.sbatch",
            "freeze_replay_scan_adapter_debug.sbatch",
            "freeze_replay_source_inventory_debug.sbatch",
            "freeze_modern_mix_recipes_debug.sbatch",
            "freeze_reused_validation_panels_debug.sbatch",
            "inspect_native_overlap_schema_debug.sbatch",
            "inspect_hf_release_debug.sbatch",
            "inspect_replay_parquet_debug.sbatch",
            "materialize_replay_postfilter_scan_debug.sbatch",
            "prepare_greek_source_stream_debug.sbatch",
            "split_replay_stage_b_debug.sbatch",
            "tokenize_h2g_stream_debug.sbatch",
            "tokenize_phase3_stream_debug.sbatch",
            "export_realized_document_ledger_debug.sbatch",
            "run_fresh_greekmmlu_stream_scan_debug.sbatch",
            "run_native_suite_replay_scan_debug.sbatch",
            "filter_replay_greekmmlu_debug.sbatch",
            "audit_replay_stage_b_greekmmlu_debug.sbatch",
            "materialize_phase3_openarchives_candidates_debug.sbatch",
            "prepare_phase3_openarchives_candidates_debug.sbatch",
            "anonymize_phase3_openarchives_candidates_debug.sbatch",
            "build_phase3_openarchives_catalog_debug.sbatch",
            "build_phase3_unseen_catalog_debug.sbatch",
            "freeze_greekmmlu_sentinels_debug.sbatch",
            "validate_greekmmlu_sentinels_debug.sbatch",
        )
        stale = "/iopsstor/scratch/cscs/fffoivos/python_envs/greek_cpt25b_v1_py312/bin/python"
        for name in wrappers:
            text = (ROOT / "clariden" / name).read_text(encoding="utf-8")
            self.assertIn("H2G_DATA_PYTHON:?set verified bundle-bound data runtime Python", text)
            self.assertIn("H2G_DATA_RUNTIME_ROOT:?set verified bundle-bound data runtime root", text)
            self.assertTrue(
                "verify_data_runtime.py" in text or "verify_data_runtime.inc.sh" in text,
                f"{name} does not execute the bundle-bound runtime verifier",
            )
            if "verify_data_runtime.inc.sh" in text:
                self.assertLess(
                    text.index("verify_code_bundle.py"),
                    text.index("verify_data_runtime.inc.sh"),
                    f"{name} sources the runtime verifier before authenticating its code bundle",
                )
            self.assertNotIn(stale, text)

    def test_native_suite_replay_scan_uses_bundle_data_runtime(self) -> None:
        text = (ROOT / "clariden" / "run_native_suite_replay_scan_debug.sbatch").read_text(encoding="utf-8")
        self.assertIn('uenv run pytorch/v2.9.1:v2 --view=default --', text)
        self.assertIn('"$data_python" "$scanner"', text)
        self.assertIn('"$data_python" "$subproject/scripts/finalize_native_training_exclusions.py"', text)
        self.assertNotIn("pytorch/v2.6.0:v1", text)

    def test_cross_bundle_adoption_is_explicit_and_narrow(self) -> None:
        freezer = (ROOT / "scripts/freeze_producer_bundle_compatibility.py").read_text(encoding="utf-8")
        data = (ROOT / "scripts/freeze_pre_main_data_authorities.py").read_text(encoding="utf-8")
        legacy = (ROOT / "scripts/freeze_legacy_public_evaluator.py").read_text(encoding="utf-8")
        wrapper = (ROOT / "clariden/freeze_producer_bundle_compatibility_debug.sbatch").read_text(encoding="utf-8")
        self.assertIn("ALLOWED_CHANGED_PATHS", freezer)
        self.assertIn("producer bundle has unaudited changed paths", freezer)
        self.assertIn(
            '"frozen_training_tools/bakeoff_training/bakeoff_train.sbatch"',
            freezer,
        )
        self.assertIn("all_bundle_receipts_fully_reverified", freezer)
        self.assertIn("--producer-compatibility", data)
        self.assertIn("require_accepted_producer", data)
        self.assertIn("--producer-compatibility", legacy)
        self.assertIn("verify_code_bundle.py", wrapper)
        self.assertIn("H2G_PRODUCER_BUNDLE_RECEIPTS:?", wrapper)

    def test_sentinel_freeze_is_transactional_and_cross_scale_authorized(self) -> None:
        freeze = (ROOT / "clariden/freeze_greekmmlu_sentinels_debug.sbatch").read_text(encoding="utf-8")
        authority = (ROOT / "scripts/freeze_cross_scale_sentinel_authority.py").read_text(encoding="utf-8")
        self.assertIn(".greekmmlu_sentinels.${SLURM_JOB_ID}.partial", freeze)
        self.assertIn('mv "$temporary" "$root"', freeze)
        self.assertIn('"early": [0, 238, 476, 714]', authority)
        self.assertIn('"late": [2618, 2856, 3094, 3218]', authority)
        self.assertIn('"8b": validate_calibration', authority)
        self.assertIn('"1p5b": validate_calibration', authority)
        self.assertEqual(
            ROLE_SCHEMAS["same_stack_sentinel_calibration_state"],
            "apertus_greekmmlu_sentinel_calibration_authority_v1",
        )

    def test_replay_filters_precede_stage_b_and_post_audits_bind_stage_b_bytes(self) -> None:
        stage_b = (ROOT / "clariden/anonymize_training_stream_debug.sbatch").read_text(encoding="utf-8")
        self.assertIn('input="$H2G_STAGE_ROOT/data/greekmmlu_scans/replay_selected/clean.jsonl"', stage_b)
        self.assertIn('input_receipt="$H2G_STAGE_ROOT/receipts/greekmmlu_scan_replay_selected.json"', stage_b)
        self.assertNotIn('input="$H2G_STAGE_ROOT/data/replay_native_clean.jsonl"', stage_b)

        authority = (ROOT / "scripts/freeze_pre_main_data_authorities.py").read_text(encoding="utf-8")
        self.assertIn('replay_greek.get("input") == binding_without_counts(replay_native["output"])', authority)
        self.assertIn('replay_stage.get("input") == binding_without_counts(replay_greek["clean"])', authority)

        pre_filter = (ROOT / "clariden/filter_replay_greekmmlu_debug.sbatch").read_text(encoding="utf-8")
        self.assertIn("--stream-name replay_selected", pre_filter)
        self.assertIn('replay_native_clean.jsonl"', pre_filter)
        self.assertNotIn("--audit-only", pre_filter)

        audit = (ROOT / "clariden/audit_replay_stage_b_greekmmlu_debug.sbatch").read_text(encoding="utf-8")
        self.assertIn("--stream-name replay_selected_post", audit)
        self.assertIn('replay_selected_stage_b.jsonl"', audit)
        self.assertIn("--audit-only", audit)

        materialize = (ROOT / "clariden/materialize_replay_postfilter_scan_debug.sbatch").read_text(encoding="utf-8")
        self.assertIn('input="$H2G_STAGE_ROOT/data/replay_selected_stage_b.jsonl"', materialize)
        self.assertIn('input_receipt="$H2G_STAGE_ROOT/receipts/replay_selected_stage_b.json"', materialize)
        self.assertIn("--zero-greekmmlu-receipt", materialize)
        self.assertNotIn("greekmmlu_scans/replay_selected/clean.jsonl", materialize)

    def test_replay_inventory_publishes_authority_receipt_last(self) -> None:
        text = (ROOT / "clariden/freeze_replay_source_inventory_debug.sbatch").read_text(encoding="utf-8")
        self.assertIn("--replay-acquisition-receipt", text)
        self.assertIn("temporary_root=", text)
        recipe_publish = text.index('mv "$temporary_recipes" "$recipe_root"')
        derived_publish = text.index('mv "$temporary_derived" "$derived_recipe"')
        receipt_publish = text.index('mv "$temporary_receipt" "$inventory_receipt"')
        self.assertLess(recipe_publish, derived_publish)
        self.assertLess(derived_publish, receipt_publish)

    def test_constant_floor_overrides_restored_scheduler_and_param_group(self) -> None:
        class Scheduler:
            init_lr = 1.0e-6
            max_lr = 5.5e-5
            min_lr = 5.5e-6
            lr_warmup_steps = 400
            lr_decay_style = "WSD"

        scheduler = Scheduler()
        group = {"max_lr": 5.5e-5, "min_lr": 5.5e-6, "lr": 9.0e-6, "lr_mult": 0.5}
        self.assertTrue(enforce_constant_floor(scheduler, group, 5.5e-6))
        self.assertEqual((scheduler.init_lr, scheduler.max_lr, scheduler.min_lr), (5.5e-6, 5.5e-6, 5.5e-6))
        self.assertEqual((scheduler.lr_warmup_steps, scheduler.lr_decay_style), (0, "constant"))
        self.assertEqual((group["max_lr"], group["min_lr"], group["lr"]), (5.5e-6, 5.5e-6, 2.75e-6))
        self.assertFalse(enforce_constant_floor(scheduler, group, 5.5e-6))

    def test_scale_geometry_patch_is_exact_and_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            trainer = Path(temporary) / "bakeoff_train.sbatch"
            trainer.write_text(OLD_NETWORK + "middle\n" + OLD_METADATA + OLD_DATA_THREADS, encoding="utf-8")
            self.assertTrue(patch_trainer(trainer))
            text = trainer.read_text(encoding="utf-8")
            self.assertEqual(text.count(NEW_NETWORK), 1)
            self.assertEqual(text.count(NEW_METADATA), 1)
            self.assertEqual(text.count(NEW_DATA_THREADS), 1)
            self.assertNotIn(OLD_NETWORK, text)
            self.assertFalse(patch_trainer(trainer))

    def test_training_wrapper_uses_outer_graceful_stop_and_frozen_trainer(self) -> None:
        text = (ROOT / "clariden/train_hard_h_to_g_segment.sbatch").read_text(encoding="utf-8")
        self.assertIn("#SBATCH --signal=B:USR1@600", text)
        self.assertIn("trap request_graceful_stop USR1 TERM INT", text)
        self.assertIn("$H2G_CODE_ROOT/frozen_training_tools/bakeoff_training", text)
        self.assertIn("--megatron-receipt", text)
        self.assertIn("--phase-cache-tree-sha256", text)
        self.assertIn("--phase-cache-root", text)
        self.assertIn("--source-phase-cache-receipt", text)
        self.assertIn("--training-run-permit", text)
        self.assertNotIn("\nsbatch ", text)
        self.assertNotIn("\nscancel ", text)

    def test_phase_boundary_checkpoint_permit_binds_source_not_target_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "checkpoints/iter_0002261"
            checkpoint.mkdir(parents=True)
            (checkpoint.parent / "latest_checkpointed_iteration.txt").write_text("2261\n", encoding="utf-8")
            source_receipt = root / "phase1.json"
            source_receipt.write_text("{}\n", encoding="utf-8")
            bundle = {"root": "/immutable/code", "tree_sha256": "a" * 64}
            permit = {
                "schema_version": "apertus_hard_h_to_g_checkpoint_permit_v2",
                "status": "passed",
                "scale": "8b",
                "source_phase": 1,
                "update": 2261,
                "checkpoint_root": str(checkpoint),
                "load_root": str(checkpoint.parent),
                "load_tracker": checkpoint_permits.file_binding(checkpoint.parent / "latest_checkpointed_iteration.txt"),
                "source_phase_cache_receipt": checkpoint_permits.file_binding(source_receipt),
                "checks": {name: True for name in checkpoint_permits.REQUIRED_CHECKS},
                "executing_code_bundle": bundle,
            }
            with mock.patch.object(checkpoint_permits, "executing_code_bundle", return_value=bundle):
                checkpoint_permits.validate_permit(
                    permit,
                    scale="8b",
                    source_phase=1,
                    update=2261,
                    checkpoint_root=checkpoint,
                    source_phase_cache_receipt=source_receipt,
                )
                with self.assertRaisesRegex(ValueError, "source-phase drift"):
                    checkpoint_permits.validate_permit(
                        permit,
                        scale="8b",
                        source_phase=2,
                        update=2261,
                        checkpoint_root=checkpoint,
                        source_phase_cache_receipt=source_receipt,
                    )

    def test_training_run_permit_binds_exact_profile_and_lr(self) -> None:
        bundle = {"root": "/immutable/code", "tree_sha256": "b" * 64}
        permit = {
            "schema_version": "apertus_hard_h_to_g_training_run_permit_v1",
            "status": "passed",
            "scale": "8b",
            "profile": {
                "nodes": 16,
                "tensor_parallel": 2,
                "microbatch": 2,
                "global_batch_sequences": 1024,
            },
            "learning_rate": {"peak": "5.5e-5", "floor": "5.5e-6"},
            "executing_code_bundle": bundle,
        }
        with mock.patch.object(training_run_permits, "executing_code_bundle", return_value=bundle):
            training_run_permits.validate_permit(
                permit,
                scale="8b",
                nodes=16,
                tensor_parallel=2,
                microbatch=2,
                peak_lr="5.5e-5",
                floor_lr="5.5e-6",
            )
            with self.assertRaisesRegex(ValueError, "profile drift"):
                training_run_permits.validate_permit(
                    permit,
                    scale="8b",
                    nodes=8,
                    tensor_parallel=2,
                    microbatch=2,
                    peak_lr="5.5e-5",
                    floor_lr="5.5e-6",
                )

    def test_checkpoint_log_audit_rejects_nan_or_skipped_updates(self) -> None:
        template = (
            "iteration {iteration}/ 3694 | consumed samples: {samples} | "
            "lm loss: 1.25E+00 | grad norm: 0.42 | params norm: 7000.0 | "
            "number of skipped iterations: {skipped} | number of nan iterations: {nan} |\n"
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "train.out"
            path.write_text(
                template.format(iteration=3456, samples=3456 * 1024, skipped=0, nan=0)
                + "successfully saved checkpoint from iteration    3456 to /checkpoints\n",
                encoding="utf-8",
            )
            summary = parse_training_log(path, 3456)
            self.assertEqual(summary["last_update"], 3456)
            path.write_text(
                template.format(iteration=3456, samples=3456 * 1024, skipped=0, nan=1)
                + "successfully saved checkpoint from iteration    3456 to /checkpoints\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "skipped or NaN"):
                parse_training_log(path, 3456)

    def test_executable_document_boundary_flags_match_machine_contract(self) -> None:
        text = (ROOT / "clariden/hard_h_to_g_training.env").read_text(encoding="utf-8")
        self.assertIn('CROSS_DOC_ATTN_FLAGS="--reset-attention-mask --reset-position-ids"', text)
        self.assertIn('EOD_LOSS_MASK_FLAG="--eod-mask-loss"', text)
        training = self.experiment["training"]
        self.assertFalse(training["cross_document_attention"])
        self.assertTrue(training["attention_mask_reset_at_document_boundary"])
        self.assertTrue(training["position_reset_at_document_boundary"])
        self.assertTrue(training["eod_loss_masking"])

    def test_executable_phase_cursor_uses_strict_cache_bound_guard(self) -> None:
        text = (ROOT / "clariden/hard_h_to_g_training.env").read_text(encoding="utf-8")
        self.assertIn("phase_local_data_index_guard.py", text)
        self.assertNotIn('TRAINER_WRAPPER="$H2G_CODE_ROOT/frozen_training_tools/phase_relative_data_index.py"', text)
        for name in (
            "PHASE_LOCAL_DATA_INDEX",
            "EXPECTED_GLOBAL_UPDATE",
            "EXPECTED_PHASE_LOCAL_CONSUMED_SAMPLES",
            "EXPECTED_DATA_CACHE_SHA256",
            "ACTUAL_DATA_CACHE_SHA256",
        ):
            self.assertIn(name, text)
        constant_floor = (ROOT / "scripts/constant_floor_resume.py").read_text(encoding="utf-8")
        self.assertIn('"phase_local_data_index_guard.py"', constant_floor)
        self.assertNotIn('"phase_relative_data_index.py"', constant_floor)

    def test_one_pass_greek_preparation_composes_exact_frozen_functions(self) -> None:
        repo = ROOT.parents[1]
        e001_path = repo / "subprojects/05_token_distillation_cpt/03_training_experiments/dataset_build/hplt_clean.py"
        scanner_path = repo / "subprojects/05_token_distillation_cpt/02_corpus_preparation/30_decontaminate/scripts/decontaminate.py"
        e001 = prepared_greek.load_module("test_h2g_e001", e001_path)
        scanner = prepared_greek.load_module("test_h2g_scanner", scanner_path)
        queries = [{
            "benchmark": "greekmmlu", "example_id": "q1", "subject": "test",
            "question": "ποια είναι η πρωτεύουσα της ελλάδας σήμερα παρακαλώ",
            "choices": ["Αθήνα", "Πάτρα"], "answer_index": 0,
        }]
        items, _ = scanner.build_item_grams(queries, 8, 0.5)
        index = scanner.build_global_q_index(items)
        prepared_greek._STATE.update(e001=e001, scanner=scanner, items=items, index=index, k=8)
        text = "ποια είναι η πρωτεύουσα της ελλάδας σήμερα παρακαλώ \u200b Αθήνα"
        line, excluded, categories, removed, changed = prepared_greek.process_row((
            "data/part.parquet", 7,
            {"text": text, "source_dataset": "s", "source_doc_id": "d", "source_metadata_json": "{}"},
        ))
        output = json.loads(line)
        expected_clean, expected_removed = e001.clean_text(text)
        self.assertEqual(output["text"], expected_clean)
        self.assertEqual(removed, expected_removed)
        self.assertTrue(changed)
        self.assertTrue(excluded)
        self.assertEqual(categories, [(0, "q_plus_correct_only")])
        self.assertEqual(output["release_shard"], "data/part.parquet")
        self.assertEqual(output["release_row_index"], 7)

    def test_legacy_public_evaluator_contract_is_exact(self) -> None:
        value = read_json(ROOT / "configs/legacy_public_greekmmlu_v1.json")
        validate_legacy_contract(copy.deepcopy(value))
        value["invocation"]["dtype"] = "float32"
        with self.assertRaisesRegex(ValueError, "dtype drift"):
            validate_legacy_contract(value)

    def test_legacy_evaluator_role_requires_loader_only_snapshot_parity(self) -> None:
        receipt = {
            "schema_version": ROLE_SCHEMAS["legacy_public_evaluator_contract"],
            "loader_change_scope": "dataset_loading_only",
            "snapshot_query_receipt": {},
            "snapshot": {},
            "loader_parity_receipt": {},
            "snapshot_adapter": {},
            "code_revision": "cfdd0e7b00761a736be660867bf3d09733e24a92",
            "clean_panel_is_scientific_primary": True,
        }
        self.assertTrue(role_semantics_match("legacy_public_evaluator_contract", receipt))
        receipt["loader_change_scope"] = "scoring_and_loading"
        self.assertFalse(role_semantics_match("legacy_public_evaluator_contract", receipt))

    def test_legacy_snapshot_adapter_is_the_only_scoring_wrapper_change(self) -> None:
        parity = (ROOT / "scripts/freeze_legacy_greekmmlu_loader_parity.py").read_text(encoding="utf-8")
        adapter = (ROOT / "scripts/run_legacy_greekmmlu_snapshot_eval.py").read_text(encoding="utf-8")
        wrapper = (ROOT / "clariden/freeze_greekmmlu_queries_debug.sbatch").read_text(encoding="utf-8")
        self.assertIn("all_raw_row_hashes_match_pinned_revision", parity)
        self.assertIn("legacy._load_dataset = snapshot_loader", adapter)
        self.assertIn("freeze_legacy_greekmmlu_loader_parity.py", wrapper)

    def test_statistical_contract_uses_first_differences(self) -> None:
        decisions = validate_statistics(copy.deepcopy(self.experiment))
        goal_b = decisions["goal_b"]
        self.assertTrue(goal_b["cumulative_improvement_correlation_forbidden"])
        self.assertEqual(goal_b["immediate_switch_pair"], [2261, 2380])
        value = copy.deepcopy(self.experiment)
        value["statistics"]["trajectory_correlation"] = "cumulative_from_initialization"
        with self.assertRaisesRegex(ValueError, "trajectory correlation drift"):
            validate_statistics(value)

    def test_regenerated_query_contract_has_fixed_timestamp_and_revision(self) -> None:
        value = read_json(ROOT / "configs/greekmmlu_query_regeneration_v1.json")
        self.assertEqual(value["builder_arguments"]["generated_utc"], "2026-08-14T00:00:00Z")
        self.assertEqual(value["dataset"]["revision"], self.experiment["data"]["greekmmlu"]["revision"])
        wrapper = (ROOT / "clariden/freeze_greekmmlu_queries_debug.sbatch").read_text(encoding="utf-8")
        self.assertIn("build_frozen_greekmmlu_queries.py", wrapper)
        self.assertNotIn("--benchmarks", wrapper)
        self.assertIn("native_greek_eval_py312_aug2026", wrapper)
        self.assertIn("import datasets,pyarrow", wrapper)
        native_wrapper = (ROOT / "clariden/freeze_native_suite_scan_authority_debug.sbatch").read_text(
            encoding="utf-8"
        )
        self.assertIn("--frozen-examples-jsonl", native_wrapper)
        self.assertIn("--benchmarks all", native_wrapper)

    def test_data_runtime_is_immutable_debug_built_and_executable_verified(self) -> None:
        builder = (ROOT / "clariden/build_data_runtime_debug.sbatch").read_text(encoding="utf-8")
        validation = (ROOT / "clariden/validate_and_inspect_debug.sbatch").read_text(encoding="utf-8")
        verifier = (ROOT / "scripts/verify_data_runtime.py").read_text(encoding="utf-8")
        self.assertIn("#SBATCH --partition=debug", builder)
        self.assertIn('[[ ! -e "$TARGET8_RUNTIME_ROOT" ]]', builder)
        self.assertIn("pyarrow.compute", builder)
        self.assertIn("chmod -R a-w", builder)
        self.assertIn("quarantine_failed_runtime", builder)
        self.assertIn(
            "uenv run pytorch/v2.9.1:v2 --view=default -- env PYTHONDONTWRITEBYTECODE=1",
            builder,
        )
        self.assertIn("TARGET8_DATA_RUNTIME_ROOT:?", validation)
        self.assertIn("verify_data_runtime.py", validation)
        self.assertNotIn("greek_cpt25b_v1_py312", validation)
        self.assertIn("configs/data_runtime_requirements_v1.txt", builder)
        self.assertIn("configs/data_runtime_requirements_v1.txt", validation)

        expected_lock = (
            "pyarrow==21.0.0\n"
            "tokenizers==0.22.1\n"
            "transformers==4.57.0\n"
            "huggingface-hub==0.36.0\n"
            "datasets==4.0.0\n"
            "numpy==2.5.1\n"
            "duckdb==1.5.4\n"
            "blake3==1.0.9\n"
            "regex==2026.7.10\n"
            "zstandard==0.25.0\n"
            "pytest==8.4.1\n"
        )
        self.assertEqual(
            (ROOT / "configs/data_runtime_requirements_v1.txt").read_text(encoding="utf-8"),
            expected_lock,
        )
        self.assertIn('SCHEMA = "apertus_hard_h_to_g_data_runtime_v1"', verifier)
        self.assertIn('"pyarrow.compute"', verifier)
        self.assertIn('"transformers": "4.57.0"', verifier)
        self.assertIn('"torch_version"', verifier)

    def test_training_dcp_compatibility_check_runs_in_pinned_uenv(self) -> None:
        wrapper = (ROOT / "clariden/train_hard_h_to_g_segment.sbatch").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            "uenv run pytorch/v2.9.1:v2 --view=default -- \\\n"
            '  env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$runtime_compat_dir"',
            wrapper,
        )
        self.assertNotIn(
            'env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$runtime_compat_dir" \\\n'
            "  python3",
            wrapper,
        )

    def test_1p5b_td_authorization_has_self_contained_canonical_digest(self) -> None:
        freezer = (ROOT / "scripts/freeze_1p5b_td_policy_authorization.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("def digest(value: Any) -> str:", freezer)
        self.assertNotIn("from contract_utils import digest", freezer)

    def test_long_td_scan_has_receipt_bound_cpu_only_xfer_route(self) -> None:
        builder = (ROOT / "clariden/build_td_xfer_runtime.sbatch").read_text(encoding="utf-8")
        scan = (ROOT / "clariden/build_td_snippets_xfer.sbatch").read_text(encoding="utf-8")
        verifier = (ROOT / "scripts/verify_td_xfer_runtime.py").read_text(encoding="utf-8")
        requirements = (ROOT / "configs/td_xfer_runtime_requirements_v1.txt").read_text(
            encoding="utf-8"
        )
        self.assertIn("#SBATCH --partition=xfer", builder)
        self.assertIn("#SBATCH --partition=xfer", scan)
        self.assertIn('[[ -z "${SLURM_JOB_GPUS:-}" ]]', builder)
        self.assertIn('[[ -z "${SLURM_JOB_GPUS:-}" ]]', scan)
        self.assertIn("verify_td_xfer_runtime.py", builder)
        self.assertIn("verify_td_xfer_runtime.py", scan)
        self.assertIn("td_coverage_prepass_batched.py", scan)
        self.assertIn("--target-extended-tokens 2000000000", scan)
        self.assertIn("--parity-documents 256", scan)
        self.assertIn("td_training_inputs.json", scan)
        self.assertIn('SCHEMA = "apertus_hard_h_to_g_td_xfer_runtime_v1"', verifier)
        self.assertEqual(requirements.splitlines(), ["numpy==2.4.6", "tokenizers==0.22.1"])

    def test_evaluation_contract_group_is_transactional_and_producer_authority_versionable(self) -> None:
        evaluation = (ROOT / "clariden/freeze_hard_h_to_g_evaluation_contracts_debug.sbatch").read_text(
            encoding="utf-8"
        )
        compatibility = (ROOT / "clariden/freeze_producer_bundle_compatibility_debug.sbatch").read_text(
            encoding="utf-8"
        )
        self.assertIn(".evaluation_contracts.${SLURM_JOB_ID}.partial", evaluation)
        self.assertIn("_failed_evaluation_contracts.${SLURM_JOB_ID}", evaluation)
        self.assertIn("preserve_failed_group", evaluation)
        self.assertIn('for name in "${names[@]}"; do mv', evaluation)
        self.assertIn("H2G_PRODUCER_COMPATIBILITY_OUTPUT", compatibility)

    def test_greekmmlu_sentinel_loader_consumes_bound_clean_id_file(self) -> None:
        loader = (ROOT / "evaluation/freeze_greekmmlu_examples.py").read_text(encoding="utf-8")
        self.assertIn('require_file_binding(value.get("clean_example_ids"))', loader)
        self.assertIn('value.get("clean_count") == 16_159', loader)
        self.assertIn('value.get("dataset_revision") == REVISION', loader)

    def test_online_validation_inventory_uses_historical_ext_suffix(self) -> None:
        inventory = (ROOT / "scripts/freeze_online_validation_binaries.py").read_text(encoding="utf-8")
        self.assertIn('f"{stem}_ext_text_document"', inventory)
        self.assertNotIn('_extended_text_document"', inventory)


if __name__ == "__main__":
    unittest.main()
