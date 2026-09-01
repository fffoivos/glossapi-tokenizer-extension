from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "initialization" / "select_tied_td_pilot.py"


def load_module():
    spec = importlib.util.spec_from_file_location("select_tied_td_pilot", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


class TiedTDPilotSelectionTests(unittest.TestCase):
    def build_case(self, root: Path, candidate_bpb: float, baseline_bpb: float) -> list[str]:
        pilot_root = root / "pilots"
        fvt_model = root / "fvt_model"
        for name in ("config.json", "model.safetensors", "tied_retok_manifest.json", "tokenizer.json"):
            path = fvt_model / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((name + "\n").encode())
        ids = root / "pilot_token_ids.txt"
        ids.write_text("".join(f"{131072 + index}\n" for index in range(1024)), encoding="utf-8")
        import hashlib

        ids_sha = hashlib.sha256(ids.read_bytes()).hexdigest()
        selection = root / "pilot_token_selection.json"
        write_json(
            selection,
            {
                "schema_version": "apertus_mini_td_pilot_token_selection_v1",
                "status": "frozen",
                "selected_count": 1024,
                "modern_selected": 768,
                "polytonic_selected": 256,
                "token_ids_file": {"sha256": ids_sha},
            },
        )
        candidates = {
            "layer7_mse": (7, "mse"),
            "layer7_mse_ce_auto": (7, "mse_ce_auto"),
            "last_mse": (-1, "mse"),
            "last_mse_ce_auto": (-1, "mse_ce_auto"),
        }
        for index, (pilot_id, (layer, loss)) in enumerate(candidates.items()):
            candidate = pilot_root / pilot_id
            write_json(
                candidate / "tied_td_manifest.json",
                {
                    "status": "completed",
                    "scope": "pilot",
                    "requested_token_count": 1024,
                    "trained_token_count": 1024,
                    "trained_token_fraction": 1.0,
                    "target_layer": layer,
                    "loss_profile": loss,
                    "token_ids_file_sha256": ids_sha,
                    "input_output_share_storage": True,
                    "norm_collapse_gate_passed": True,
                },
            )
            write_json(
                candidate / "initialization_verification.json",
                {"status": "pass", "checks": {"all": True}},
            )
            for slice_name in ("hplt", "non_hplt", "polytonic"):
                write_json(
                    candidate / "metrics" / slice_name / "tokenizer_fair_metrics.json",
                    {
                        "model_path": str(candidate),
                        "global": {"bpb_bits_per_byte": candidate_bpb + index * 0.01},
                    },
                )
        baseline = root / "fvt"
        for slice_name in ("hplt", "non_hplt", "polytonic"):
            write_json(
                baseline / "metrics" / slice_name / "tokenizer_fair_metrics.json",
                {
                    "model_path": str(fvt_model),
                    "global": {"bpb_bits_per_byte": baseline_bpb},
                },
            )
        return [
            str(SCRIPT),
            "--pilot-root",
            str(pilot_root),
            "--pilot-token-selection",
            str(selection),
            "--fvt-baseline-dir",
            str(baseline),
            "--fvt-model-dir",
            str(fvt_model),
            "--output",
            str(root / "selected.json"),
        ]

    def test_accepts_best_recipe_when_it_beats_fvt(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            argv = self.build_case(root, candidate_bpb=1.9, baseline_bpb=2.0)
            with mock.patch.object(sys, "argv", argv):
                self.assertEqual(module.main(), 0)
            result = json.loads((root / "selected.json").read_text())
            self.assertEqual(result["selected_pilot_id"], "layer7_mse")
            self.assertTrue(result["fvt_macro_non_regression_gate_passed"])
            self.assertLess(result["selected_delta_macro_bpb_vs_fvt"], 0)

    def test_rejects_all_regressing_recipes(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            argv = self.build_case(root, candidate_bpb=2.1, baseline_bpb=2.0)
            with mock.patch.object(sys, "argv", argv):
                with self.assertRaisesRegex(ValueError, "all tied-TD pilots regress"):
                    module.main()


if __name__ == "__main__":
    unittest.main()
