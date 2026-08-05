from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class TDPilotSelectionTests(unittest.TestCase):
    def test_shared_pilot_covers_modern_and_polytonic_stages(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            coverage = root / "coverage.jsonl"
            with coverage.open("w", encoding="utf-8") as handle:
                for token_id in range(131_072, 148_992):
                    row = {
                        "new_token_id": token_id,
                        "status": "enough_25",
                        "usable_snippets_25": 25,
                        "extended_firings": 25 + token_id % 997,
                        "docs_with_firing": 25,
                        "base_subtoken_ids": [token_id % 100, (token_id + 1) % 100],
                        "base_subtoken_len": 2,
                    }
                    handle.write(json.dumps(row) + "\n")
            output = root / "selection"
            subprocess.run(
                [
                    "python3",
                    str(ROOT / "initialization" / "select_mini_td_pilot_tokens.py"),
                    "--coverage-jsonl",
                    str(coverage),
                    "--output-dir",
                    str(output),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
            )
            manifest = json.loads((output / "pilot_token_selection.json").read_text())
            ids = [int(value) for value in (output / "pilot_token_ids.txt").read_text().splitlines()]
            self.assertEqual(manifest["modern_selected"], 768)
            self.assertEqual(manifest["polytonic_selected"], 256)
            self.assertEqual(len(ids), 1_024)
            self.assertEqual(len(set(ids)), 1_024)
            self.assertEqual(sum(value >= 148_480 for value in ids), 256)

    def test_full_inventory_is_exact_and_unpadded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ids = root / "ids.txt"
            receipt = root / "receipt.json"
            subprocess.run(
                [
                    "python3",
                    str(ROOT / "initialization" / "write_full_added_token_ids.py"),
                    "--output",
                    str(ids),
                    "--receipt",
                    str(receipt),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
            )
            values = [int(value) for value in ids.read_text().splitlines()]
            self.assertEqual(values, list(range(131_072, 148_992)))
            self.assertEqual((131_072 + len(values)) % 256, 0)


if __name__ == "__main__":
    unittest.main()
