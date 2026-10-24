"""Readiness-contract fixtures only; no real models, assets or network calls."""

import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/check_install.py"
spec = importlib.util.spec_from_file_location("shoplens_install_check_fixture", SCRIPT)
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


class InstallReadinessTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="shoplens-readiness-diagnostic-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.calls = []
        self.weights_ready = True
        self.bad_mode, self.bad_contract = None, None
        self.items = {}
        self.modes = ("qwen_frozen", "qwen_semantic", "qwen_preference")
        for domain in checker.DOMAINS:
            relative = f"data/images/{domain}/DIAGNOSTIC.jpg"
            path = self.root / relative
            path.parent.mkdir(parents=True)
            path.write_bytes(b"DIAGNOSTIC file-existence fixture, not an actual image")
            self.items[domain] = [{"item_id": f"DIAGNOSTIC-{domain}", "image_path": relative}]
        def forbidden(*args, **kwargs):
            raise AssertionError("Readiness checks must not load encoders or perform query inference")
        encoders = SimpleNamespace(load_catalog=self.catalog, status=self.status,
                                   encode_query=forbidden, encode_queries=forbidden, encode_image=forbidden,
                                   _query_model=forbidden, _image_model=forbidden)
        models = SimpleNamespace(MODES=self.modes, model_status=self.model_status, personalization=self.personalization)
        self.context = patch.object(checker, "_runtime_api", return_value=(encoders, models))
        self.context.start()
        self.addCleanup(self.context.stop)

    def catalog(self, root, domain):
        self.calls.append(("catalog", domain))
        return {"items": self.items[domain], "text": SimpleNamespace(shape=(1, 1024)), "image": SimpleNamespace(shape=(1, 512))}

    def status(self, root):
        return {"ready": self.weights_ready, "query_ready": self.weights_ready, "image_ready": self.weights_ready,
                "device": "cpu", "qwen": {"ready": self.weights_ready, "model_id": "DIAGNOSTIC-QWEN", "revision": "fixture", "path": "/must/not/leak/private/path"},
                "clip": {"ready": self.weights_ready, "model_id": "DIAGNOSTIC-CLIP", "revision": "fixture"}}

    def model_status(self, root, domain):
        return {mode: {"ready": mode != self.bad_mode, "error": "DIAGNOSTIC export unavailable" if mode == self.bad_mode else None} for mode in self.modes}

    def personalization(self, root, domain, mode, positive):
        self.assertEqual(positive, [])
        self.calls.append(("verify_export", domain, mode))
        return {"scores": [] if mode == self.bad_contract else None,
                "metadata": {"active": False, "seed": 42, "selected_epoch": 1, "artifact_signature": f"diagnostic-{domain}-{mode}"}}

    def test_three_catalogs_nine_exports_and_portable_photos_are_checked_without_encoding(self):
        report = checker.check(self.root, require_weights=True)
        self.assertTrue(report["ok"], report["errors"])
        self.assertEqual(sum(call[0] == "catalog" for call in self.calls), 3)
        self.assertEqual(sum(call[0] == "verify_export" for call in self.calls), 9)
        self.assertEqual(report["domains"]["sports"]["text_shape"], [1, 1024])
        self.assertEqual(report["domains"]["sports"]["photos"]["missing_count"], 0)
        self.assertNotIn("path", report["weights"]["qwen"])
        self.assertIn("127.0.0.1", report["launch_command"])

    def test_missing_weights_warn_by_default_and_fail_when_required(self):
        self.weights_ready = False
        optional = checker.check(self.root)
        required = checker.check(self.root, require_weights=True)
        self.assertTrue(optional["ok"])
        self.assertTrue(optional["warnings"])
        self.assertFalse(required["ok"])
        self.assertTrue(any("prepare_studio_models.py" in message for message in required["errors"]))

    def test_missing_or_nonportable_photos_fail_without_loading_a_model(self):
        (self.root / self.items["sports"][0]["image_path"]).unlink()
        self.items["baby"][0]["image_path"] = str(self.root / "outside-diagnostic.jpg")
        report = checker.check(self.root)
        self.assertFalse(report["ok"])
        self.assertEqual(report["domains"]["sports"]["photos"]["missing_count"], 1)
        self.assertEqual(report["domains"]["baby"]["photos"]["unsafe_count"], 1)

    def test_unavailable_export_and_empty_history_contract_failure_are_errors(self):
        self.bad_mode = "qwen_semantic"
        unavailable = checker.check(self.root)
        self.assertFalse(unavailable["ok"])
        self.assertTrue(any("export unavailable" in message for message in unavailable["errors"]))
        self.bad_mode, self.bad_contract = None, "qwen_preference"
        bad_contract = checker.check(self.root)
        self.assertFalse(bad_contract["ok"])
        self.assertTrue(any("empty-history inference contract" in message for message in bad_contract["errors"]))

    def test_missing_dependencies_produce_actionable_install_guidance(self):
        with patch.object(checker, "_runtime_api", side_effect=ImportError("DIAGNOSTIC missing torch")):
            report = checker.check(self.root)
        self.assertFalse(report["ok"])
        self.assertIn("requirements.txt", report["errors"][0])
        self.assertEqual(self.calls, [])

    def test_older_python_fails_before_importing_runtime_dependencies(self):
        with patch.object(checker.sys, "version_info", (3, 11, 9)), patch.object(checker, "_runtime_api") as api:
            report = checker.check(self.root)
        self.assertFalse(report["ok"])
        self.assertIn("Python 3.12", report["errors"][0])
        api.assert_not_called()

    def test_cli_help_does_not_require_site_packages_or_model_imports(self):
        completed = subprocess.run([sys.executable, "-S", str(SCRIPT), "--help"], capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--require-weights", completed.stdout)
        self.assertIn("--json", completed.stdout)


if __name__ == "__main__":
    unittest.main()
