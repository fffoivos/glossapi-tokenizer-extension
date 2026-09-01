from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class GreekMMLUCleanSubsetTests(unittest.TestCase):
    def test_receipt_bound_bridge_dropped_ledgers_are_supported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            queries = root / "queries.jsonl"
            queries.write_text(
                "".join(
                    json.dumps({"example_id": f"greekmmlu:{index}"}) + "\n"
                    for index in range(16_632)
                )
            )
            dropped = root / "shard.dropped.jsonl"
            dropped.write_text(
                json.dumps(
                    {
                        "evidence": [
                            {"benchmark_item_id": "greekmmlu:7"},
                            {"benchmark_item_id": "greekmmlu:11"},
                        ]
                    }
                )
                + "\n"
            )
            import hashlib

            manifest = root / "shard.manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "schema_version": "full_cpt_megatron_shard_v1",
                        "kind": "training",
                        "decontamination": {"status": "frozen"},
                        "counts": {"contaminated_rows": 1},
                        "outputs": {
                            "dropped_ledger": {
                                "path": str(dropped),
                                "bytes": dropped.stat().st_size,
                                "sha256": hashlib.sha256(dropped.read_bytes()).hexdigest(),
                            }
                        },
                    }
                )
                + "\n"
            )
            ids = root / "clean.txt"
            receipt = root / "clean.json"
            subprocess.run(
                [
                    "python3",
                    str(ROOT / "evaluation" / "build_greekmmlu_clean_subset.py"),
                    "--queries-jsonl",
                    str(queries),
                    "--ledger-root",
                    str(root),
                    "--output-ids",
                    str(ids),
                    "--output-manifest",
                    str(receipt),
                ],
                check=True,
            )
            value = json.loads(receipt.read_text())
            self.assertEqual(value["full_count"], 16_632)
            self.assertEqual(value["contaminated_item_count"], 2)
            self.assertEqual(value["clean_count"], 16_630)
            self.assertEqual(len(value["bridge_manifests"]), 1)
            clean = set(ids.read_text().splitlines())
            self.assertNotIn("greekmmlu:7", clean)
            self.assertNotIn("greekmmlu:11", clean)


if __name__ == "__main__":
    unittest.main()
