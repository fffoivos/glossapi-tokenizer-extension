from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

EVAL_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EVAL_DIR))

from sequence_models.char_tcn_crf import encode_utf8_lines
from sequence_models.contract import (
    ContractError,
    GoldDocument,
    GoldLine,
    build_split_manifest,
    validate_gold,
    validate_silver,
)
from sequence_models.evaluate import evaluate, promotion_report, read_predictions, work_bootstrap
from sequence_models.feature_crf import LinearChainCRF, make_examples
from sequence_models.features import FeatureEncoder, TAGS, classes_to_bioes, validate_bioes
from sequence_models.runtime import compare_prediction_files
from sequence_models.silver_reconstruct import audit_tracked


def document(
    document_id: str,
    source: str = "greek_phd",
    split: str = "test",
    labels: tuple[str, ...] = ("O", "BIB", "BIB", "O"),
) -> GoldDocument:
    lines = tuple(
        GoldLine(
            f"{document_id}:{index}", index, text, label, tokens,
            None if label == "UNKNOWN" else label == "O",
        )
        for index, (text, label, tokens) in enumerate(zip(
            ("Κανονικό κείμενο", "Smith (2020). Title.", "Athens: Press.", "Τέλος"),
            labels,
            (10, 4, 3, 5),
        ))
    )
    return GoldDocument(
        document_id=document_id,
        work_id=document_id,
        representation_id=f"rep-{document_id}",
        source=source,
        split=split,
        coverage="full_document",
        n_physical_lines=4,
        n_present_lines=4,
        annotation_status="human_adjudicated" if split == "test" else "human_single",
        annotator_ids=("a", "b"),
        adjudicator_id="c" if split == "test" else None,
        tokenizer_id="tok",
        tokenizer_revision="rev",
        lines=lines,
    )


class ContractTests(unittest.TestCase):
    def test_unknown_is_explicit_and_promotion_rejects_it(self) -> None:
        doc = document("w-1", labels=("O", "UNKNOWN", "BIB", "O"))
        policy = {
            "test_annotation_status": "human_adjudicated",
            "test_coverage": "full_document",
            "required_sources": ["greek_phd"],
            "minimum_test_documents": 1,
            "minimum_test_documents_per_source": 1,
            "minimum_double_annotated_fraction": 0.0,
            "require_split_manifest": False,
        }
        with self.assertRaisesRegex(ContractError, "UNKNOWN"):
            validate_gold([doc], policy, for_promotion=True)

    def test_source_balanced_work_split_is_deterministic_and_grouped(self) -> None:
        docs = []
        for source in ("greek_phd", "openarchives.gr"):
            for index in range(20):
                docs.append(document(f"{source}-{index}", source=source, split="train"))
        policy = {
            "seed": "fixed", "train_fraction": 0.55,
            "validation_fraction": 0.15, "test_fraction": 0.30,
        }
        first = build_split_manifest(docs, policy)
        second = build_split_manifest(list(reversed(docs)), policy)
        self.assertEqual(first["assignments"], second["assignments"])
        for source in ("greek_phd", "openarchives.gr"):
            counts = {split: 0 for split in ("train", "validation", "test")}
            for doc in docs:
                if doc.source == source:
                    counts[first["assignments"][doc.document_id]] += 1
            self.assertEqual(counts, {"train": 11, "validation": 3, "test": 6})

    def test_locked_manifest_binds_document_work_and_source_inventory(self) -> None:
        doc = document("w-1", split="train")
        policy = {"test_annotation_status": "human_adjudicated", "test_coverage": "full_document"}
        manifest = {
            "schema_version": "academic-structure-split-v1",
            "inventory_sha256": "wrong",
            "assignments": {doc.document_id: "train"},
        }
        with self.assertRaisesRegex(ContractError, "inventory"):
            validate_gold([doc], policy, split_manifest=manifest)

    def test_silver_contract_is_leak_free_but_never_production_eligible(self) -> None:
        base = document("silver-1", split="train")
        silver = replace(
            base,
            annotation_status="LLM_silver",
            annotator_ids=("LLM:Claude-Opus",),
            adjudicator_id=None,
            annotation_engine="Claude Opus span-annotation workflow",
            task_scope="bibliography_binary_windows",
            coverage="annotated_windows",
            lines=tuple(replace(line, is_running_prose=None) for line in base.lines),
        )
        split_policy = {
            "seed": "fixed", "train_fraction": 1.0,
            "validation_fraction": 0.0, "test_fraction": 0.0,
        }
        manifest = build_split_manifest([silver], split_policy)
        receipt = validate_silver(
            iter([silver]),
            {
                "annotation_status": "LLM_silver",
                "allowed_task_scopes": ["bibliography_binary_windows"],
            },
            split_manifest=manifest,
        )
        self.assertEqual(receipt["evidence_tier"], "LLM_silver")
        self.assertFalse(receipt["production_eligible"])
        self.assertEqual(receipt["source_counts"], {"greek_phd": 1})


class ModelTests(unittest.TestCase):
    def test_unknown_splits_crf_training_sequences(self) -> None:
        doc = document("w-1", split="train", labels=("O", "UNKNOWN", "BIB", "O"))
        examples = make_examples([doc], FeatureEncoder(char_hash_dim=64))
        self.assertEqual([example.line_indices for example in examples], [(0,), (2, 3)])

    def test_numpy_crf_gradient_and_viterbi_are_finite_and_legal(self) -> None:
        labels = ("O", "BIB", "BIB", "O")
        tags = classes_to_bioes(labels)
        validate_bioes(tags)
        gold = np.asarray([TAGS.index(tag) for tag in tags])
        rows = [{0: 1.0}, {1: 1.0}, {1: 0.5}, {0: 1.0}]
        model = LinearChainCRF(2, seed=7)
        loss, *_ = model.nll_and_grad(rows, gold)
        self.assertTrue(math.isfinite(loss))
        decoded = [TAGS[index] for index in model.viterbi(rows)]
        validate_bioes(decoded)

    def test_utf8_encoder_preserves_ocr_bytes(self) -> None:
        encoded = encode_utf8_lines(["α", "a|pha"], max_bytes=4)
        self.assertEqual(encoded[0], [byte + 1 for byte in "α".encode("utf-8")])
        self.assertEqual(len(encoded[1]), 4)

    def test_bib_only_crf_cannot_decode_toc(self) -> None:
        model = LinearChainCRF(2, seed=7, active_classes=("BIB",))
        rows = [{0: 1.0}, {1: 1.0}, {0: 1.0}]
        decoded = [TAGS[index] for index in model.viterbi(rows)]
        self.assertFalse(any(tag.endswith("-TOC") for tag in decoded))
        self.assertTrue(all(not allowed for tag, allowed in zip(TAGS, model.active_tag_mask)
                            if tag.endswith("-TOC")))


class ReconstructionAuditTests(unittest.TestCase):
    def test_tracked_inventory_proves_current_text_dependency(self) -> None:
        report = audit_tracked()
        evidence = report["sequence_evidence"]
        self.assertEqual(evidence["annotated_windows"], 3339)
        self.assertEqual(evidence["documents"], 1738)
        self.assertEqual(evidence["bibliography_span_annotations"], 3186)
        self.assertEqual(evidence["missing_text_batch_file_count"], 240)
        self.assertEqual(evidence["fit_ready_line_rows"], 0)
        self.assertFalse(evidence["toc_supervision_available"])
        self.assertFalse(report["production_eligible"])


class EvaluationTests(unittest.TestCase):
    def test_metrics_mask_unknown_and_measure_true_retention(self) -> None:
        doc = document("w-1", labels=("O", "UNKNOWN", "BIB", "O"))
        metrics, per_doc = evaluate([doc], {doc.document_id: ["O", "BIB", "BIB", "O"]})
        self.assertEqual(metrics["token"]["action_precision"], 1.0)
        self.assertEqual(metrics["token"]["true_main_text_retention"], 1.0)
        self.assertEqual(metrics["coverage"]["unknown_lines"], 1)
        self.assertEqual(metrics["document"]["catastrophic_prose_deletions"], 0)
        ci = work_bootstrap([doc], per_doc, per_doc, replicates=20, seed=3)
        self.assertEqual(ci["bib_recall_gain"], [0.0, 0.0])

    def test_document_maximum_is_not_summed_across_work_clusters(self) -> None:
        left, right = document("left"), document("right")
        predictions = {
            left.document_id: ["BIB", "BIB", "BIB", "O"],
            right.document_id: ["BIB", "BIB", "BIB", "O"],
        }
        metrics, _ = evaluate([left, right], predictions)
        self.assertEqual(metrics["document"]["maximum_contiguous_false_deletion_tokens"], 10)

    def test_prediction_join_is_identity_strict_and_parity_hook_is_exact(self) -> None:
        doc = document("w-1")
        row = {
            "schema_version": "academic-structure-predictions-v1",
            "model_id": "x", "document_id": doc.document_id,
            "lines": [
                {"line_id": line.line_id, "abs_idx": line.abs_idx, "prediction": label}
                for line, label in zip(doc.lines, ("O", "BIB", "BIB", "O"))
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            left = Path(directory) / "left.jsonl"
            right = Path(directory) / "right.jsonl"
            left.write_text(json.dumps(row) + "\n", encoding="utf-8")
            right.write_text(json.dumps(row) + "\n", encoding="utf-8")
            self.assertEqual(read_predictions(left, [doc])[doc.document_id][1], "BIB")
            self.assertEqual(compare_prediction_files(left, right)["status"], "pass")

    def test_promotion_gate_requires_every_safety_and_recall_check(self) -> None:
        candidate = {
            "token": {"action_precision": 0.991, "prose_contamination": 0.0005,
                      "bib_recall": 0.90, "toc_recall": 0.80},
            "document": {"catastrophic_prose_deletions": 0},
            "by_source": {"s": {"token": {"action_precision": 0.99, "bib_recall": 0.90, "toc_recall": 0.80}}},
        }
        baseline = {
            "token": {"bib_recall": 0.80, "toc_recall": 0.70},
            "by_source": {"s": {"token": {"bib_recall": 0.80, "toc_recall": 0.70}}},
        }
        confidence = {"action_precision": [0.986, 0.995], "prose_contamination": [0.0, 0.0015]}
        gates = {
            "minimum_deletion_token_precision": 0.990,
            "minimum_deletion_token_precision_ci95_lower": 0.985,
            "minimum_per_source_deletion_token_precision": 0.980,
            "maximum_prose_token_contamination": 0.001,
            "maximum_prose_token_contamination_ci95_upper": 0.002,
            "minimum_bib_recall_gain": 0.03,
            "minimum_toc_recall_gain": 0.05,
            "maximum_per_head_recall_regression": 0.01,
            "maximum_per_source_recall_regression": 0.03,
            "pergamos_allows_recall_regression": False,
        }
        gates.update({
            "maximum_artifact_bytes": 1000,
            "maximum_peak_rss_bytes": 1000,
            "maximum_cpu_hours_relative_to_c0": 5.0,
            "maximum_allowlist_wall_hours_one_cpu_node": 24.0,
            "require_python_runtime_parity": True,
        })
        report = promotion_report(candidate, baseline, confidence, gates)
        self.assertEqual(report["statistical_status"], "pass")
        self.assertEqual(report["status"], "blocked")
        report = promotion_report(
            candidate, baseline, confidence, gates,
            artifact_bytes=10,
            runtime_parity={"status": "pass"},
            candidate_benchmark={"best_seconds": 2.0, "peak_rss_bytes": 100},
            baseline_benchmark={"best_seconds": 1.0},
        )
        self.assertEqual(report["status"], "pass")
        candidate["document"]["catastrophic_prose_deletions"] = 1
        self.assertEqual(promotion_report(candidate, baseline, confidence, gates)["status"], "fail")


if __name__ == "__main__":
    unittest.main()
