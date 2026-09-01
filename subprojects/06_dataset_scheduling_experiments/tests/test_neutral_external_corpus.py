from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def receipt(path: Path, **extra) -> dict:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        **extra,
    }


class NeutralExternalCorpusTests(unittest.TestCase):
    def test_freeze_requires_source_separation_and_exact_minhash_cross_dedup(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_snapshot = root / "snapshot.json"
            source_snapshot.write_text('{"snapshot":"post-cutoff"}\n')
            candidate = root / "candidate.jsonl"
            rows = [
                {"cluster_id": "c1", "source_id": "external-a", "text": "Πρώτο ανεξάρτητο κείμενο."},
                {"cluster_id": "c2", "source_id": "external-b", "text": "Δεύτερο ανεξάρτητο κείμενο."},
            ]
            candidate.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
            pool = root / "pool.json"
            pool.write_text(
                json.dumps(
                    {
                        "schema_version": "apertus_mini_schedule_pool_corpus_v1",
                        "status": "completed",
                        "global_identity_proof": {"modern_greek_exact_content_duplicates_or_collisions": 0},
                    }
                )
            )
            dedup = root / "dedup.json"
            dedup.write_text(
                json.dumps(
                    {
                        "schema_version": "apertus_mini_neutral_external_dedup_v1",
                        "status": "passed",
                        "training_reference": {"pool_corpus_receipt": receipt(pool)},
                        "dataset_separation": {
                            "document_cluster_split": True,
                            "publishers_or_domains_absent_from_training": True,
                            "source_time_window_absent_from_training": False,
                            "candidate_documents_never_used_for_training": True,
                            "evaluation_use_authorized": True,
                        },
                        "exact_dedup": {
                            "algorithm": "sha256_utf8_text",
                            "candidate_internal_duplicate_rows": 0,
                            "candidate_to_training_match_rows": 0,
                        },
                        "minhash_dedup": {
                            "token_shingle_size": 5,
                            "permutations": 128,
                            "bands": 32,
                            "rows_per_band": 4,
                            "threshold": 0.85,
                            "candidate_internal_pairs_at_or_above_threshold": 0,
                            "candidate_to_training_pairs_at_or_above_threshold": 0,
                        },
                        "source_snapshot_receipts": [receipt(source_snapshot)],
                        "candidate_output": receipt(candidate, rows=2),
                    }
                )
            )
            output = root / "frozen.json"
            subprocess.run(
                [
                    "python3",
                    str(ROOT / "evaluation" / "finalize_neutral_external_corpus.py"),
                    "--dedup-receipt",
                    str(dedup),
                    "--pool-corpus-receipt",
                    str(pool),
                    "--output",
                    str(output),
                ],
                check=True,
                stdout=subprocess.PIPE,
                text=True,
            )
            frozen = json.loads(output.read_text())
            self.assertEqual(frozen["documents"], 2)
            self.assertTrue(frozen["global_minhash_dedup_against_training"])
            self.assertEqual(frozen["minhash_threshold"], 0.85)
            self.assertTrue(frozen["publishers_or_domains_absent_from_training"])
            self.assertFalse(frozen["source_time_window_absent_from_training"])

            rejected = json.loads(dedup.read_text())
            rejected["dataset_separation"]["publishers_or_domains_absent_from_training"] = False
            dedup.write_text(json.dumps(rejected))
            with self.assertRaises(subprocess.CalledProcessError):
                subprocess.run(
                    [
                        "python3",
                        str(ROOT / "evaluation" / "finalize_neutral_external_corpus.py"),
                        "--dedup-receipt",
                        str(dedup),
                        "--pool-corpus-receipt",
                        str(pool),
                        "--output",
                        str(root / "must_not_freeze.json"),
                    ],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )


if __name__ == "__main__":
    unittest.main()
