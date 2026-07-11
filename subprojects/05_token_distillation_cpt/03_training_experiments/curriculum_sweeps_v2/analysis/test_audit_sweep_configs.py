import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import audit_sweep_configs as audit


MANIFEST = HERE.parent / "results" / "sweep_config_audit_20260711.json"
COMMON_ENV = HERE.parents[1] / "configs" / "common_cpt.env"
TRAIN_DIR = HERE.parent / "train"


class SweepConfigAuditTests(unittest.TestCase):
    def test_checked_in_manifest_policy(self):
        manifest = json.loads(MANIFEST.read_text())
        audit.validate_manifest(manifest)
        self.assertEqual(manifest["canonical"]["ademamix_beta2"], 0.999)
        self.assertEqual(manifest["canonical"]["lr_warmup_iters"], 400)

    def test_canonical_env_defaults(self):
        script = (
            f"source {COMMON_ENV!s}; "
            "printf '%s %s %s %s %s %s' "
            '"$ADEMA_BETA2" "$ADEMA_BETA3" "$ADEMA_ALPHA" '
            '"$LR_PEAK" "$LR_WARMUP_ITERS" "$LR_WSD_DECAY_SAMPLES"'
        )
        env = {"PATH": os.environ["PATH"]}
        values = subprocess.check_output(["bash", "-c", script], env=env, text=True).split()
        self.assertEqual(values, ["0.999", "0.999", "4.0", "5.5e-5", "400", "659179"])

    def test_historical_sweep_launchers_pin_as_run_beta2_and_warmup(self):
        for name in [
            "sweep_replay.sh",
            "sweep_peak_lr.sh",
            "sweep_alpha.sh",
            "sweep_beta3.sh",
            "submit_vanilla_control.sh",
        ]:
            text = (TRAIN_DIR / name).read_text()
            self.assertRegex(text, r"ADEMA_BETA2=.*0\.995", name)
            self.assertRegex(text, r"LR_WARMUP_ITERS=.*400", name)

    def _fixture(self, drift=False):
        temp = tempfile.TemporaryDirectory()
        root = Path(temp.name)
        runs = []
        fingerprints = []
        for tag, beta2 in [("b2_099", 0.99), ("b2_0995", 0.995), ("b2_0999", 0.999)]:
            metadata = {
                "ademamix_beta2": beta2,
                "ademamix_beta3": 0.999,
                "ademamix_alpha": 4.0,
                "lr_warmup_samples": 409600,
                "data_seed": 7 if not drift or tag != "b2_0999" else 8,
                "output_dir": str(root / tag),
                "start_time": tag,
            }
            run_dir = root / tag
            (run_dir / "checkpoints").mkdir(parents=True)
            raw = (json.dumps(metadata, indent=2) + "\n").encode()
            (run_dir / "run_metadata.json").write_bytes(raw)
            (run_dir / "checkpoints" / "latest_checkpointed_iteration.txt").write_text("3218")
            fingerprints.append(audit.fingerprint(metadata, {"ademamix_beta2"}))
            runs.append(
                {
                    "run_tag": tag,
                    "metadata_sha256": hashlib.sha256(raw).hexdigest(),
                    "varied": {"ademamix_beta2": beta2},
                    "final_checkpoint": 3218,
                }
            )
        manifest = {
            "schema_version": 1,
            "canonical": {"ademamix_beta2": 0.999, "lr_warmup_iters": 400},
            "beta2_policy": {
                "selected_if_comparable": 0.999,
                "fallback": 0.995,
                "fixed_lr_warmup_iters": 400,
            },
            "sweeps": {
                "beta2": {
                    "varied_fields": ["ademamix_beta2"],
                    "comparable": not drift,
                    "normalized_common_sha256": fingerprints[0],
                    "required_common": {"lr_warmup_samples": 409600},
                    "runs": runs,
                }
            },
        }
        return temp, root, manifest

    def test_live_audit_accepts_numeric_equivalence(self):
        temp, root, manifest = self._fixture()
        self.addCleanup(temp.cleanup)
        audit.validate_manifest(manifest)
        self.assertEqual(audit.audit_live(manifest, root), ["beta2: 3 mechanically comparable runs"])

    def test_live_audit_rejects_hidden_drift(self):
        temp, root, manifest = self._fixture(drift=True)
        self.addCleanup(temp.cleanup)
        manifest["sweeps"]["beta2"]["comparable"] = True
        audit.validate_manifest(manifest)
        with self.assertRaisesRegex(audit.AuditError, "comparability"):
            audit.audit_live(manifest, root)

    def test_policy_falls_back_when_beta2_is_not_comparable(self):
        temp, _, manifest = self._fixture(drift=True)
        self.addCleanup(temp.cleanup)
        with self.assertRaisesRegex(audit.AuditError, "beta2 policy"):
            audit.validate_manifest(manifest)
        manifest["canonical"]["ademamix_beta2"] = 0.995
        audit.validate_manifest(manifest)


if __name__ == "__main__":
    unittest.main()
