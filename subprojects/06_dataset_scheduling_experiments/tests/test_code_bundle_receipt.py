from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class CodeBundleReceiptTests(unittest.TestCase):
    def test_verifier_runs_by_absolute_path_with_safe_path_enabled(self) -> None:
        result = subprocess.run(
            [
                "python3",
                "-I",
                "-B",
                str(ROOT / "production" / "verify_code_bundle.py"),
                "--help",
            ],
            cwd="/",
            env={"PYTHONSAFEPATH": "1"},
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_frozen_bundle_rejects_later_file_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = root / "bundle"
            bundle.mkdir()
            source = bundle / "worker.py"
            source.write_text("print('v1')\n")
            receipt = root / "receipt.json"
            subprocess.run(
                [
                    "python3",
                    str(ROOT / "production" / "freeze_code_bundle.py"),
                    "--root",
                    str(bundle),
                    "--kind",
                    "scientific",
                    "--output",
                    str(receipt),
                ],
                check=True,
                stdout=subprocess.PIPE,
                text=True,
            )
            payload = json.loads(receipt.read_text())
            self.assertEqual(payload["file_count"], 1)
            source.write_text("print('v2')\n")
            result = subprocess.run(
                [
                    "python3",
                    "-c",
                    (
                        "from pathlib import Path; "
                        "from production.campaign_contract import verify_code_bundle_receipt; "
                        f"verify_code_bundle_receipt(Path({str(receipt)!r}), Path({str(bundle)!r}), 'scientific')"
                    ),
                ],
                cwd=ROOT,
                check=False,
                stderr=subprocess.PIPE,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("bundle file drift", result.stderr)


if __name__ == "__main__":
    unittest.main()
