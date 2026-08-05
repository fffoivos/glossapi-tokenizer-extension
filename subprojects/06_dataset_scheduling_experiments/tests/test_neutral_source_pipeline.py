from __future__ import annotations

import importlib.util
import csv
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SOURCE = load_module(
    "prepare_greek_parliament_neutral_source",
    ROOT / "evaluation" / "prepare_greek_parliament_neutral_source.py",
)
CROSS = load_module(
    "cross_deduplicate_neutral_external",
    ROOT / "evaluation" / "cross_deduplicate_neutral_external.py",
)


class NeutralSourcePipelineTests(unittest.TestCase):
    def test_cluster_identity_is_sitting_level_and_stable(self) -> None:
        first = {
            "sitting_date": " 12/01/2000 ",
            "parliamentary_period": "Ι",
            "parliamentary_session": "Α",
            "parliamentary_sitting": "  42 ",
            "speech": "one",
        }
        same_sitting = {**first, "speech": "different speech"}
        next_sitting = {**first, "parliamentary_sitting": "43"}
        self.assertEqual(SOURCE.cluster_id(first), SOURCE.cluster_id(same_sitting))
        self.assertNotEqual(SOURCE.cluster_id(first), SOURCE.cluster_id(next_sitting))

    def test_source_reader_accepts_speeches_larger_than_default_csv_limit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive_path = Path(directory) / "sample.zip"
            csv_path = Path(directory) / "Greek_Parliament_Proceedings_sample.csv"
            with csv_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["speech", "sitting_date", "parliamentary_sitting"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "speech": "α" * 200_000,
                        "sitting_date": "2000-01-12",
                        "parliamentary_sitting": "42",
                    }
                )
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.write(csv_path, arcname=csv_path.name)
            rows = list(SOURCE.iter_rows(archive_path, csv_path.name))
            self.assertEqual(len(rows), 1)
            self.assertEqual(len(rows[0][1]["speech"]), 200_000)

    def test_sorted_signature_groups_and_search_are_exact(self) -> None:
        record_dtype, signature_dtype = CROSS._signature_dtypes()
        records = np.array(
            [
                (1, 2, 3, 4, 11),
                (1, 2, 3, 4, 12),
                (5, 6, 7, 8, 13),
            ],
            dtype=record_dtype,
        )
        signatures = CROSS._signature_only(records, signature_dtype)
        unique, starts, ends = CROSS._candidate_groups(records, signatures)
        self.assertEqual(starts.tolist(), [0, 2])
        self.assertEqual(ends.tolist(), [2, 3])
        self.assertEqual(np.searchsorted(signatures, unique, side="left").tolist(), [0, 2])
        self.assertEqual(np.searchsorted(signatures, unique, side="right").tolist(), [2, 3])

    def test_empty_shingle_sets_are_not_duplicate_evidence(self) -> None:
        empty = np.array([], dtype=np.uint64)
        nonempty = np.array([1, 2, 3], dtype=np.uint64)
        self.assertEqual(CROSS._jaccard(empty, empty), 0.0)
        self.assertEqual(CROSS._jaccard(empty, nonempty), 0.0)
        self.assertEqual(CROSS._jaccard(nonempty, nonempty), 1.0)

    def test_internal_dedup_compares_only_cluster_aggregates(self) -> None:
        by_doc = {
            1: {"doc_type": "speech_fragment"},
            2: {"doc_type": "complete_sitting"},
            3: {"doc_type": "speech_fragment"},
            4: {"doc_type": "complete_sitting"},
        }
        self.assertEqual(CROSS._aggregate_doc_ids([1, 2, 3, 4], by_doc), [2, 4])


if __name__ == "__main__":
    unittest.main()
