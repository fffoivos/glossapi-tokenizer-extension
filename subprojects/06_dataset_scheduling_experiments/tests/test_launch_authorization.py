from __future__ import annotations

import json
import hashlib
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class LaunchAuthorizationTests(unittest.TestCase):
    def _write_gate_receipt(
        self,
        path: Path,
        evidence: Path,
        gate_id: str,
        *,
        include_semantic_proof: bool = True,
    ) -> None:
        matrix = path.parent / "matrix.json"
        validator = ROOT / "production" / "finalize_launch_gate_set.py"
        payload = {
            "schema_version": "apertus_mini_launch_gate_receipt_v1",
            "status": "passed",
            "launch_authorized": True,
            "gate_id": gate_id,
            "experiment_matrix": {
                "path": str(matrix.resolve()),
                "bytes": matrix.stat().st_size,
                "sha256": hashlib.sha256(matrix.read_bytes()).hexdigest(),
            },
            "evidence": [{
                "path": str(evidence),
                "bytes": evidence.stat().st_size,
                "sha256": hashlib.sha256(evidence.read_bytes()).hexdigest(),
            }],
        }
        if include_semantic_proof:
            payload["semantic_validation"] = {
                "schema_version": "apertus_mini_launch_gate_semantics_v1",
                "validator": "production/finalize_launch_gate_set.py",
                "validator_sha256": hashlib.sha256(validator.read_bytes()).hexdigest(),
                "all_gate_specific_checks_passed": True,
            }
        path.write_text(json.dumps(payload))

    def test_authorization_rejects_missing_declared_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            matrix = root / "matrix.json"
            matrix.write_text(
                json.dumps(
                    {
                        "launch_authorized": False,
                        "launch_gates": ["gate_a", "gate_b"],
                    }
                )
            )
            evidence = root / "evidence.json"
            evidence.write_text("{}")
            receipt = root / "gate_a.json"
            self._write_gate_receipt(receipt, evidence, "gate_a")
            result = subprocess.run(
                [
                    "python3",
                    str(ROOT / "production" / "authorize_experiment_matrix.py"),
                    "--experiment-matrix",
                    str(matrix),
                    "--gate-receipt",
                    str(receipt),
                    "--output",
                    str(root / "authorized.json"),
                ],
                check=False,
                text=True,
                stderr=subprocess.PIPE,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("missing=['gate_b']", result.stderr)

    def test_authorization_rejects_generic_receipt_without_semantic_proof(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            matrix = root / "matrix.json"
            matrix.write_text(json.dumps({"launch_authorized": False, "launch_gates": ["gate_a"]}))
            evidence = root / "evidence.json"
            evidence.write_text("{}")
            receipt = root / "gate_a.json"
            self._write_gate_receipt(
                receipt,
                evidence,
                "gate_a",
                include_semantic_proof=False,
            )
            result = subprocess.run(
                [
                    "python3",
                    str(ROOT / "production" / "authorize_experiment_matrix.py"),
                    "--experiment-matrix",
                    str(matrix),
                    "--gate-receipt",
                    str(receipt),
                    "--output",
                    str(root / "authorized.json"),
                ],
                check=False,
                text=True,
                stderr=subprocess.PIPE,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("lacks the semantic validator proof", result.stderr)


if __name__ == "__main__":
    unittest.main()
