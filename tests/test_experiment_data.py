"""Integrity checks for presenting existing evidence, with no model execution."""

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from shoplens.experiment_data import ExperimentDataError, load_experiment_summary, load_training_traces

SOURCE_ROOT = Path(__file__).resolve().parents[1]


class ExperimentDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source_bytes = (SOURCE_ROOT / "results/qwen/results_qwen_summary.json").read_bytes()
        cls.report = json.loads(cls.source_bytes)

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.path = self.root / "results/qwen/results_qwen_summary.json"
        self.path.parent.mkdir(parents=True)
        self.path.write_bytes(self.source_bytes)

    def change(self, edit):
        report = copy.deepcopy(self.report)
        edit(report)
        self.path.write_text(json.dumps(report), encoding="utf-8")

    def test_real_snapshot_retains_reported_metrics_and_hash(self):
        result = load_experiment_summary(self.root)
        self.assertEqual(result["results"], self.report["results"])
        self.assertEqual(result["paired_comparisons"], self.report["paired_comparisons"])
        self.assertEqual(result["provenance"]["source_sha256"], hashlib.sha256(self.source_bytes).hexdigest())
        self.assertEqual(len(result["training"]), 9)
        self.assertEqual(len(result["training_all"]), 27)
        self.assertEqual(len(result["trace_rows"]), 54)
        self.assertEqual(len(result["paired_rows"]), 27)
        self.assertFalse(result["provenance"]["live_training_or_full_export_audit_repeated"])

    def test_incomplete_or_failed_historical_audit_is_rejected(self):
        edits = [lambda x: x.update(complete=False),
                 lambda x: x["verification"].update(complete=False),
                 lambda x: x["verification"].update(errors=[{"check": "failure"}]),
                 lambda x: x["verification"].update(missing_units=["baby/qwen_frozen/seed42"]),
                 lambda x: x["verification"].update(observed_identities=26)]
        for edit in edits:
            with self.subTest(edit=edit):
                self.change(edit)
                with self.assertRaises(ExperimentDataError):
                    load_experiment_summary(self.root)

    def test_duplicate_groups_and_incorrect_seed_coverage_are_rejected(self):
        for edit in (lambda x: x["results"].__setitem__(1, x["results"][0]),
                     lambda x: x["results"][0].update(seeds=[42, 42, 43]),
                     lambda x: x["verification"]["unit_evidence"].pop("baby/qwen_frozen/seed44")):
            with self.subTest(edit=edit):
                self.change(edit)
                with self.assertRaises(ExperimentDataError):
                    load_experiment_summary(self.root)

    def test_malformed_baseline_preservation_has_a_controlled_error(self):
        for value in (None, [], "invalid"):
            with self.subTest(value=value):
                self.change(lambda x: x["verification"].update(baseline_preservation=value))
                with self.assertRaisesRegex(ExperimentDataError, "baseline preservation"):
                    load_experiment_summary(self.root)

    def test_nonfinite_metric_or_negative_sd_is_rejected(self):
        for value in (float("nan"), float("inf"), -.01):
            with self.subTest(value=value):
                self.change(lambda x: x["results"][0].update({"ndcg@10_std": value}))
                with self.assertRaises(ExperimentDataError):
                    load_experiment_summary(self.root)

    def test_paired_statistics_and_same_seed_cycle_must_be_consistent(self):
        for field, value in (("mean_difference", 0), ("std_difference", 0), ("wins", 3), ("seeds", [42, 43])):
            with self.subTest(field=field):
                self.change(lambda x: x["paired_comparisons"][0].update({field: value}))
                with self.assertRaises(ExperimentDataError):
                    load_experiment_summary(self.root)
        # Reordering the three values preserves mean/SD/counts but breaks same-seed identities.
        self.change(lambda x: x["paired_comparisons"][0]["differences"].reverse())
        with self.assertRaisesRegex(ExperimentDataError, "same-seed"):
            load_experiment_summary(self.root)

    def test_validation_checkpoint_selection_cannot_be_fabricated(self):
        unit = "baby/qwen_frozen/seed42"
        self.change(lambda x: x["verification"]["unit_evidence"][unit].update(selected_epoch=2))
        with self.assertRaisesRegex(ExperimentDataError, "Selected epoch"):
            load_experiment_summary(self.root)

    def test_curves_are_exact_recorded_validation_values(self):
        result = load_experiment_summary(self.root)
        for trace in result["training_all"]:
            unit = f"{trace['domain']}/{trace['model']}/seed{trace['seed']}"
            expected = self.report["verification"]["unit_evidence"][unit]
            self.assertEqual(trace["validation_ndcg"], expected["adaptation_validation_ndcg@10"])
            self.assertEqual(trace["selected_epoch"], expected["selected_epoch"])

    def test_unknown_private_fields_are_not_exported(self):
        self.change(lambda x: x.update(private_host="/Users/example/private", execution_recovery={"host": "secret"}))
        result = json.dumps(load_experiment_summary(self.root))
        self.assertNotIn("/Users/", result)
        self.assertNotIn("secret", result)
        self.assertNotIn("scheduler-job", result)

    def test_no_missing_summary_synthesis_and_only_explicit_fallback(self):
        self.path.unlink()
        with self.assertRaises(FileNotFoundError):
            load_experiment_summary(self.root)
        runtime = self.root / "runtime"
        (runtime / "deliverables").mkdir(parents=True)
        (runtime / "deliverables/results_qwen_summary.json").write_bytes(self.source_bytes)
        with self.assertRaises(FileNotFoundError):
            load_experiment_summary(self.root)
        result = load_experiment_summary(self.root, runtime)
        self.assertEqual(result["provenance"]["source_relative"], "deliverables/results_qwen_summary.json")

    def test_duplicate_json_keys_are_rejected(self):
        self.path.write_text('{"complete":true,"complete":false}', encoding="utf-8")
        with self.assertRaisesRegex(ExperimentDataError, "Duplicate JSON"):
            load_experiment_summary(self.root)

    def test_runtime_training_helper_does_not_synthesize_missing_traces(self):
        self.assertEqual(load_training_traces(self.root), [])
        destination = self.root / "artifacts/qwen/baby/qwen_frozen/seed42/training.json"
        destination.parent.mkdir(parents=True)
        destination.write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ExperimentDataError, "partial"):
            load_training_traces(self.root)


if __name__ == "__main__":
    unittest.main()
