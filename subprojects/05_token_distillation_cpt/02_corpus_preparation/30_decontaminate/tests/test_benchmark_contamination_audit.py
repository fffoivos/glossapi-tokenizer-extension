from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "scripts"


def load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


builder = load("benchmark_query_builder", "build_decontamination_queries.py")
scanner = load("benchmark_contamination_scanner", "audit_benchmark_contamination_parquet.py")
publisher = load("benchmark_contamination_publisher", "publish_benchmark_contamination_audit.py")
sampler = load("benchmark_contamination_sampler", "sample_benchmark_contamination_evidence.py")
finalizer = load("benchmark_contamination_finalizer", "finalize_benchmark_contamination_audit.py")
adjudication = load(
    "benchmark_contamination_adjudication",
    "build_benchmark_contamination_adjudication_packet.py",
)


class QueryBuilderTests(unittest.TestCase):
    def test_oyxoy_nli_uses_source_pair_and_groups_three_decisions(self) -> None:
        row = {
            "benchmark": "oyxoy_nli",
            "example_id": "gold:7:Unknown",
            "group_id": "gold:7",
            "question": (
                "Πρόταση αναφοράς:\nΗ πρώτη ανθρώπινη πρόταση.\n\n"
                "Υπόθεση:\nΗ δεύτερη ανθρώπινη πρόταση.\n\n"
                "Ισχύει η σχέση «ουδέτερη σχέση»;"
            ),
            "choices": ["Όχι", "Ναι"],
            "answer_index": 0,
            "subject": "Unknown",
            "metadata": {},
        }
        query = builder.query_from_frozen_row(row)
        self.assertEqual(query["question"], "Η πρώτη ανθρώπινη πρόταση.")
        self.assertEqual(query["choices"], ["Η δεύτερη ανθρώπινη πρόταση."])
        self.assertEqual(query["evaluation_unit_id"], "gold:7")
        self.assertEqual(
            query["discount_example_ids"],
            ["gold:7:Unknown", "gold:7:Entailment", "gold:7:Contradiction"],
        )
        self.assertNotIn("Ισχύει η σχέση", query["question"])


class PublisherTests(unittest.TestCase):
    def test_complete_upload_set_includes_manifest_itself(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "publish_manifest.json"
            manifest.write_text('{"status":"passed"}\n', encoding="utf-8")
            files = {"audit_receipt.json": {"bytes": 3, "sha256": "abc"}}
            payload = publisher.build_payload_files(manifest, files)
            self.assertEqual(set(payload), {"audit_receipt.json", "publish_manifest.json"})
            self.assertEqual(payload["publish_manifest.json"]["bytes"], manifest.stat().st_size)
            self.assertEqual(payload["publish_manifest.json"]["sha256"], publisher.sha256(manifest))


class ReviewSamplerTests(unittest.TestCase):
    def test_stable_rank_is_deterministic_and_uses_document_locator(self) -> None:
        row = {
            "benchmark": "demosqa",
            "evaluation_unit_id": "7",
            "source_dataset": "demo",
            "source_doc_id": "doc-1",
            "dataset_shard": "data/000001.parquet",
            "dataset_row_index": 9,
        }
        self.assertEqual(sampler.stable_rank(row), sampler.stable_rank(dict(row)))
        changed = dict(row, dataset_row_index=10)
        self.assertNotEqual(sampler.stable_rank(row), sampler.stable_rank(changed))


class FinalizerTests(unittest.TestCase):
    def test_jsonl_reader_preserves_unicode_line_separator_inside_string(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "queries.jsonl"
            rows = [{"text": "before\u2028after"}, {"text": "second"}]
            path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                encoding="utf-8",
            )
            self.assertEqual(finalizer.read_jsonl(path), rows)


class AdjudicationPacketTests(unittest.TestCase):
    def test_rank_changes_with_source_locator(self) -> None:
        row = {
            "benchmark": "gpcr",
            "evaluation_unit_id": "2",
            "source_dataset": "source",
            "source_doc_id": "doc-a",
            "dataset_shard": "data/000002.parquet",
            "dataset_row_index": 4,
        }
        self.assertEqual(adjudication.rank(row), adjudication.rank(dict(row)))
        self.assertNotEqual(adjudication.rank(row), adjudication.rank(dict(row, source_doc_id="doc-b")))


class ScannerTests(unittest.TestCase):
    def build_index(self, query: dict):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "queries.jsonl"
            path.write_text(json.dumps(query, ensure_ascii=False) + "\n", encoding="utf-8")
            return scanner.load_queries(path, k=8, minimum_short_question_tokens=3)[:2]

    def test_short_question_fallback_requires_nearby_correct_answer(self) -> None:
        query = {
            "schema": "greek-benchmark-decontam-query-v2",
            "benchmark": "medical_mcqa",
            "example_id": "16",
            "evaluation_unit_id": "16",
            "discount_example_ids": ["16"],
            "source_group_id": None,
            "query_kind": "mcq_question_correct_answer",
            "subject": "anatomy",
            "question": "Ο πνευμονικός σύνδεσμος",
            "choices": ["λάθος απάντηση", "αποτελεί ανάκαμψη των δύο πετάλων"],
            "answer_index": 1,
            "metadata": {},
        }
        queries, index = self.build_index(query)
        result = scanner.scan_document(
            {
                "source_dataset": "medical-book",
                "source_doc_id": "doc-1",
                "text": "Κεφάλαιο\nΟ πνευμονικός σύνδεσμος\nαποτελεί ανάκαμψη των δύο πετάλων.",
                "source_metadata_json": '{"url":"https://example.test/doc-1"}',
            },
            shard_path="data/000001.parquet",
            shard_row_index=4,
            queries=queries,
            pattern_index=index,
            k=8,
            max_gap_tokens=50,
            max_gap_tokens_short=5,
        )
        self.assertEqual(len(result), 1)
        self.assertTrue(result[0]["recommended_exclusion"])
        self.assertEqual(result[0]["question_pattern_kind"], "short_exact")
        self.assertEqual(result[0]["question_line_start"], 2)
        self.assertEqual(result[0]["correct_answer_line_start"], 3)

        answer_only = scanner.scan_document(
            {"source_dataset": "x", "source_doc_id": "y", "text": "αποτελεί ανάκαμψη των δύο πετάλων."},
            shard_path="data/000001.parquet",
            shard_row_index=5,
            queries=queries,
            pattern_index=index,
            k=8,
            max_gap_tokens=50,
            max_gap_tokens_short=5,
        )
        self.assertEqual(answer_only, [])

    def test_lexical_source_pair_can_match_definition_before_usage(self) -> None:
        usage = "Του κόλλησαν την αβανιά πως τάχα αυτός ήταν ο κλέφτης"
        definition = "άδικη κατηγορία συκοφαντία κακολογία"
        query = {
            "schema": "greek-benchmark-decontam-query-v2",
            "benchmark": "oyxoy_wsd_definition",
            "example_id": "0:0:0",
            "evaluation_unit_id": "0:0:0",
            "discount_example_ids": ["0:0:0"],
            "source_group_id": "0",
            "query_kind": "lexical_usage_correct_definition",
            "subject": "definition_selection",
            "question": usage,
            "choices": [definition, "άλλη άσχετη σημασία"],
            "answer_index": 0,
            "metadata": {},
        }
        queries, index = self.build_index(query)
        result = scanner.scan_document(
            {"source_dataset": "dictionary", "source_doc_id": "entry-0", "text": f"{definition}\n{usage}"},
            shard_path="data/000002.parquet",
            shard_row_index=0,
            queries=queries,
            pattern_index=index,
            k=8,
            max_gap_tokens=50,
            max_gap_tokens_short=5,
        )
        self.assertEqual(len(result), 1)
        self.assertTrue(result[0]["recommended_exclusion"])
        self.assertEqual(result[0]["correct_answer_line_start"], 1)
        self.assertEqual(result[0]["question_line_start"], 2)


if __name__ == "__main__":
    unittest.main()
