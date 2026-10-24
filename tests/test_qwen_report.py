"""Tiny synthetic artifact fixtures test auditing; no result is a real experiment."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/make_qwen_report.py"
SPEC = importlib.util.spec_from_file_location("make_qwen_report", SCRIPT)
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


def put_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def put_bytes(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)
    return report.sha256(path)


class QwenReportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def unit(self, domain, mode, seed):
        return self.root / "artifacts/qwen" / domain / mode / f"seed{seed}"

    def complete_fixture(self):
        for domain in report.DOMAINS:
            items_hash = put_bytes(self.root / "data/processed" / domain / "items.jsonl", b"explicit synthetic item fixture\n")
            events_hash = put_bytes(self.root / "data/processed" / domain / "events.jsonl", b"explicit synthetic event fixture\n")
            old = self.root / "artifacts" / domain
            old_hash = put_bytes(old / "features.npz", b"synthetic legacy feature bytes")
            put_json(old / "features.json", {"features_sha256": old_hash})
            base = self.root / "artifacts/qwen" / domain / "base"
            base_hash = put_bytes(base / "features.npz", b"synthetic Qwen base feature bytes")
            base_metadata = {"features_sha256": base_hash, "synthetic": False, "real_data": True,
                             "identity": {"old_features_sha256": old_hash}}
            put_json(base / "features.json", base_metadata)
            collected = []
            for seed in report.SEEDS:
                for mode in report.MODES:
                    directory = self.unit(domain, mode, seed)
                    frozen = mode == "qwen_frozen"
                    token = f"{domain}:{mode}:{seed}".encode()
                    fingerprint = __import__("hashlib").sha256(f"{domain}:{seed}".encode()).hexdigest()
                    candidate_hash = put_bytes(directory / "candidate.npz", b"frozen candidates" if frozen else token)
                    head_hash = put_bytes(directory / "models" / f"{mode}.pt", token + b" head")
                    serving_hash = put_bytes(directory / "serving.npz", token + b" serving")
                    if not frozen:
                        put_bytes(directory / "adapter/adapter_model.safetensors", token + b" adapter")
                    adapter_hash = None if frozen else report.tree_hash(directory / "adapter")
                    encoding = {"max_length": 256, "padding_side": "left", "pooling": "last valid token",
                                "normalization": "L2", "truncation": True, "review_instruction": "fixed instruction"}
                    lora = None if frozen else {"r": 8, "lora_alpha": 16, "lora_dropout": .05,
                                               "target_modules": ["q_proj", "v_proj"], "bias": "none"}
                    warmup = {"head_sha256": fingerprint, "selected_epoch": 3,
                              "max_epochs": 20, "min_epochs": 5, "patience": 3,
                              "curve": [{"epoch": i + 1, "validation_ndcg@10": value}
                                        for i, value in enumerate([.01, .02, .03, .02, .01])]}
                    metadata = {"model": mode, "seed": seed, "selected_epoch": 2, "epochs": 2,
                                "family": "qwen_candidate_lora", "embedding_dim": 128, "max_history": 20,
                                "objective": mode,
                                "model_id": report.MODEL_ID, "revision": report.REVISION,
                                "encoding_config": encoding, "lora_config": lora, "warmup": warmup,
                                "items_sha256": items_hash, "events_sha256": events_hash,
                                "base_features_sha256": base_hash, "candidate_sha256": candidate_hash,
                                "head_sha256": head_hash, "serving_sha256": serving_hash,
                                "adapter_sha256": adapter_hash, "config_sha256": fingerprint,
                                "n_training_examples": 30, "max_examples": 30000}
                    put_json(directory / "models" / f"{mode}.json", metadata)
                    training = {"model": mode, "seed": seed, "selected_epoch": 2, "fingerprint": fingerprint,
                                "warmup": warmup, "sample_hashes": {"adaptation_cases_sha256": fingerprint},
                                "sampling_sha256": fingerprint, "negative_sha256_by_epoch": [fingerprint] * 2,
                                "case_order_sha256_by_epoch": [fingerprint] * 2,
                                "adaptation_validation_ndcg@10": [.01, .02],
                                "adaptation_curve": [{"epoch": i + 1, "bpr_loss": .6 - .1 * i,
                                                      "semantic_loss": 1. if mode == "qwen_semantic" else 0.,
                                                      "validation_ndcg@10": value} for i, value in enumerate([.01, .02])],
                                "model_id": report.MODEL_ID, "revision": report.REVISION,
                                "encoding_config": encoding, "lora_config": lora,
                                "selected_candidate_sha256": candidate_hash,
                                "selected_serving_sha256": serving_hash, "selected_adapter_sha256": adapter_hash,
                                "sample_metadata": {"sampled_adaptation_cases": 30, "eligible_review_cases": 30},
                                "gradient_evidence": {"base_parameters_frozen": True, "head_gradient_max_norm": .1,
                                    "total_lora_gradient_max_norm": 0. if frozen else .02,
                                    "bpr_to_lora_max_norm": .01 if mode == "qwen_preference" else 0.,
                                    "trainable_lora_parameters": 0 if frozen else 16, "gradient_probes": 0 if frozen else 2},
                                "live_cache_parity": {"candidate_max_abs_error": 1e-6, "history_score_max_abs_error": 0.,
                                                      "candidate_tolerance": 1e-5, "history_tolerance": 1e-5}}
                    put_json(directory / "training.json", training)
                    i = report.SEEDS.index(seed)
                    delta = [0., 0., 0.] if frozen else [.02, -.01, .02] if mode == "qwen_semantic" else [-.005, .002, .001]
                    score = .10 + .001 * i + delta[i]
                    metrics = {"recall@10": score * 1.5, "ndcg@10": score,
                               "recall@20": score * 2, "ndcg@20": score + .01}
                    row = {"model": mode, "seed": seed, "split": "test", "selected_epoch": 2,
                           **metrics, "n_users": 8, "n_targets": 12,
                           "items_sha256": items_hash, "events_sha256": events_hash,
                           "featurehash": base_hash, "selected_features_sha256": candidate_hash,
                           "head_sha256": head_hash, "serving_sha256": serving_hash, "adapter_sha256": adapter_hash,
                           "data_audit": {"synthetic": False}, "features_metadata": base_metadata,
                           "training_configuration": {name: metadata[name] for name in
                               ("epochs", "model_id", "revision", "encoding_config", "lora_config", "config_sha256", "objective", "warmup",
                                "n_training_examples", "max_examples")},
                           "run_protocol": {**report.EXPECTED_PROTOCOL, "eligible_targets": 12, "n_catalog_items": 4},
                           "low_interaction": {"max_training_positive_users": 5, "n_users": 3, "n_targets": 4,
                                               **{key: value * .2 for key, value in metrics.items()}}}
                    put_json(directory / "results.json", {"results": [row]})
                    put_json(directory / "complete.json", {"fingerprint": fingerprint, "selected_epoch": 2,
                             "model": mode, "seed": seed,
                             "head_sha256": head_hash, "serving_sha256": serving_hash,
                             "candidate_sha256": candidate_hash, "adapter_sha256": adapter_hash,
                             "results_sha256": report.sha256(directory / "results.json")})
                    collected.append(row)
            put_json(self.root / "artifacts/qwen" / domain / "results.json", {"results": collected})
        put_json(self.root / "experiments/qwen/baseline_verification.json", {"unchanged": True, "matched_files": 3})

    def test_empty_root_is_explicitly_incomplete(self):
        summary = report.build_report(self.root)
        self.assertFalse(summary["complete"])
        self.assertEqual(summary["raw_record_count"], 0)
        self.assertEqual(len(summary["verification"]["missing_units"]), 27)
        self.assertIn("未完成", report.render_report(summary))
        self.assertIn("INCOMPLETE", report.render_report(summary, english=True))

    def test_all_27_fixtures_pass_including_actual_file_hash_checks(self):
        self.complete_fixture()
        summary = report.build_report(self.root, verify_cache_files=True)
        self.assertEqual(summary["verification"]["errors"], [])
        self.assertTrue(summary["complete"])
        self.assertEqual(len(summary["results"]), 9)
        self.assertEqual(summary["raw_record_count"], 27)
        self.assertTrue(summary["verification"]["local_cache_bytes_verified"])

    def test_paired_differences_use_matched_seeds(self):
        self.complete_fixture()
        summary = report.build_report(self.root)
        comparison = next(row for row in summary["paired_comparisons"]
                          if row["domain"] == "baby" and row["before"] == "qwen_frozen" and row["after"] == "qwen_semantic")
        self.assertEqual(comparison["seeds"], [42, 43, 44])
        self.assertAlmostEqual(comparison["mean_difference"], .01)
        self.assertEqual((comparison["wins"], comparison["ties"], comparison["losses"]), (2, 0, 1))
        self.assertAlmostEqual(comparison["std_difference"], .017320508075688773)

    def change_training(self, domain="baby", mode="qwen_semantic", seed=42, **values):
        path = self.unit(domain, mode, seed) / "training.json"
        document = json.loads(path.read_text()); document.update(values); put_json(path, document)

    def test_different_shared_case_or_negative_hash_is_incomplete(self):
        self.complete_fixture()
        self.change_training(negative_sha256_by_epoch=["1" * 64, "2" * 64])
        summary = report.build_report(self.root)
        self.assertFalse(summary["complete"])
        self.assertTrue(any(issue["check"] == "paired-training-inputs" for issue in summary["verification"]["errors"]))

    def test_objectives_must_share_case_id_order_and_warmup(self):
        self.complete_fixture()
        path = self.unit("baby", "qwen_preference", 43) / "training.json"
        original = json.loads(path.read_text())
        changes = ({"sample_hashes": {"adaptation_cases_sha256": "a" * 64}},
                   {"case_order_sha256_by_epoch": ["b" * 64] * 2},
                   {"warmup": {**original["warmup"], "head_sha256": "c" * 64}})
        for change in changes:
            with self.subTest(field=next(iter(change))):
                put_json(path, {**original, **change})
                summary = report.build_report(self.root)
                self.assertFalse(summary["complete"])
                self.assertTrue(any(issue["check"] == "paired-training-inputs" for issue in summary["verification"]["errors"]))

    def test_protocol_mismatch_is_rejected(self):
        self.complete_fixture()
        path = self.root / "artifacts/qwen/baby/results.json"
        document = json.loads(path.read_text())
        document["results"][0]["run_protocol"]["max_history"] = 10
        put_json(path, document)
        summary = report.build_report(self.root)
        self.assertFalse(summary["complete"])
        self.assertTrue(any(issue["check"] == "declared-protocol" for issue in summary["verification"]["errors"]))

    def test_semantic_bpr_to_lora_route_is_rejected(self):
        self.complete_fixture()
        path = self.unit("baby", "qwen_semantic", 42) / "training.json"
        original = json.loads(path.read_text())
        original["gradient_evidence"]["bpr_to_lora_max_norm"] = .01
        put_json(path, original)
        summary = report.build_report(self.root)
        self.assertFalse(summary["complete"])
        self.assertTrue(any(issue["check"] == "gradient-route" for issue in summary["verification"]["errors"]))

    def test_epoch_zero_or_wrong_validation_selection_is_rejected(self):
        self.complete_fixture()
        self.change_training(selected_epoch=0)
        summary = report.build_report(self.root)
        self.assertFalse(summary["complete"])
        self.assertTrue(any(issue["check"] == "epoch-selection" for issue in summary["verification"]["errors"]))

    def test_missing_gradient_or_failed_parity_is_incomplete(self):
        self.complete_fixture()
        self.change_training(gradient_evidence={}, live_cache_parity={"candidate_max_abs_error": 1.})
        summary = report.build_report(self.root)
        self.assertFalse(summary["complete"])
        checks = {issue["check"] for issue in summary["verification"]["errors"]}
        self.assertIn("gradients", checks)
        self.assertIn("live-cache-parity", checks)

    def test_trainable_candidates_can_differ_across_seeds(self):
        self.complete_fixture()
        summary = report.build_report(self.root)
        self.assertTrue(summary["complete"])
        candidates = {summary["verification"]["unit_evidence"][f"baby/qwen_semantic/seed{seed}"]["candidate_sha256"]
                      for seed in report.SEEDS}
        self.assertEqual(len(candidates), 3)

    def test_modified_local_cache_is_rejected(self):
        self.complete_fixture()
        (self.unit("baby", "qwen_frozen", 42) / "candidate.npz").write_bytes(b"changed fixture")
        summary = report.build_report(self.root, verify_cache_files=True)
        self.assertFalse(summary["complete"])
        self.assertTrue(any(issue["check"] == "local-cache-bytes" for issue in summary["verification"]["errors"]))

    def test_diagnostic_and_partial_units_do_not_enter_aggregate_rows(self):
        self.complete_fixture()
        directory = self.unit("baby", "qwen_semantic", 42)
        for name in ("training.json", "models/qwen_semantic.json", "complete.json"):
            with self.subTest(diagnostic_document=name):
                path = directory / name
                original = path.read_bytes(); document = json.loads(original)
                document["diagnostic"] = True; put_json(path, document)
                summary = report.build_report(self.root)
                self.assertFalse(summary["complete"])
                self.assertEqual(summary["raw_record_count"], 26)
                group = next(row for row in summary["results"]
                             if row["domain"] == "baby" and row["model"] == "qwen_semantic")
                self.assertEqual(group["seeds"], [43, 44])
                path.write_bytes(original)
        (directory / "complete.json").unlink()
        summary = report.build_report(self.root)
        self.assertFalse(summary["complete"])
        self.assertEqual(summary["raw_record_count"], 26)

    def test_verification_binds_selected_json_evidence_hashes(self):
        self.complete_fixture()
        summary = report.build_report(self.root)
        hashes = summary["verification"]["unit_evidence"]["sports/qwen_semantic/seed44"]["source_evidence_sha256"]
        directory = self.unit("sports", "qwen_semantic", 44)
        self.assertEqual(set(hashes), {"training.json", "models/qwen_semantic.json", "complete.json", "results.json"})
        for name, checksum in hashes.items():
            self.assertEqual(checksum, report.sha256(directory / name))

    def test_reports_only_write_new_qwen_outputs(self):
        self.complete_fixture()
        old_report = self.root / "deliverables/实验报告.md"
        old_report.parent.mkdir(exist_ok=True);old_report.write_text("old baseline report")
        summary = report.build_report(self.root)
        report.write_reports(self.root, summary)
        self.assertEqual(old_report.read_text(), "old baseline report")
        for name in ("ShopLens_Qwen_report.md", "ShopLens_Qwen_report_en.md", "results_qwen_summary.csv", "results_qwen_summary.json"):
            self.assertTrue((self.root / "deliverables" / name).is_file())
        self.assertTrue((self.root / "experiments/qwen/verification.json").is_file())

    def recovery_note(self):
        return {"failed_job": "92956.gaas", "resumed_job": "92961.gaas",
                "affected_unit": "baby/qwen_frozen/seed42",
                "training_code_unchanged_between_jobs": True,
                "only_changed_model_module_between_jobs": "qwen_evaluate.py",
                "selected_training_checkpoints_reused": True,
                "unsaved_test_evaluation_recomputed": True,
                "test_scores_used_for_selection_or_tuning": False,
                "formal_result_policy": "One persisted result; existing saved results are reused"}

    def test_optional_recovery_note_is_embedded_and_explained_in_both_languages(self):
        self.complete_fixture()
        note = self.recovery_note()
        put_json(self.root / "experiments/qwen/recovery_note.json", note)
        summary = report.build_report(self.root, verify_cache_files=True)
        self.assertTrue(summary["complete"])
        self.assertEqual(summary["verification"]["execution_recovery"], note)
        self.assertEqual(len(summary["verification"]["execution_recovery_source_sha256"]), 64)
        zh = report.render_report(summary)
        en = report.render_report(summary, english=True)
        self.assertIn("实际运行恢复记录", zh)
        self.assertIn("未保存的评估", zh)
        self.assertIn("不代表测试计算从未重复", zh)
        self.assertIn("recomputed", en)
        self.assertIn("one persisted formal record", en)
        self.assertIn("not a claim that test computation never repeated", en)

    def test_recovery_note_cannot_make_missing_results_or_gradients_complete(self):
        put_json(self.root / "experiments/qwen/recovery_note.json", self.recovery_note())
        empty = report.build_report(self.root)
        self.assertFalse(empty["complete"])
        self.complete_fixture()
        self.change_training(gradient_evidence={})
        summary = report.build_report(self.root)
        self.assertFalse(summary["complete"])
        self.assertTrue(any(issue["check"] == "gradients" for issue in summary["verification"]["errors"]))

    def numerical_recovery_note(self, **changes):
        numerical = {"status": "complete", "failed_job": "92961.gaas",
                     "diagnostic_job": "93023.gaas", "resumed_job": "93033.gaas",
                     "affected_unit": "clothing/qwen_semantic/seed42",
                     "completed_formal_units_preserved": 19,
                     "diagnostic_first_nonfinite_step": 69,
                     "diagnostic_successful_updates_before_failure": 68,
                     "specific_fused_kernel_root_cause_established": False,
                     "failed_unit_restarted_from_initialisation": True,
                     "failed_unit_had_resume_checkpoint": False,
                     "objective_sampling_learning_rates_and_budget_unchanged": True,
                     "test_scores_used_for_selection_or_tuning": False,
                     "production_same_state_regression": "passed",
                     "production_prefix100_regression": "passed", **changes}
        note = {**self.recovery_note(), "numerical_recovery": numerical}
        put_json(self.root / "experiments/qwen/recovery_note.json", note)
        return note

    def clothing_numerics(self):
        for mode in ("qwen_semantic", "qwen_preference"):
            for seed in report.SEEDS:
                self.change_training("clothing", mode, seed, training_numerics={
                    "candidate_training_sdpa_backend": "math", "nonfinite_gradient_guard": True,
                    "cached_encoding_sdpa_backend": "automatic"})

    def test_completed_numerical_recovery_is_preserved_and_backend_limits_are_bilingual(self):
        self.complete_fixture()
        self.clothing_numerics()
        note = self.numerical_recovery_note()
        summary = report.build_report(self.root, verify_cache_files=True)
        verification = summary["verification"]
        self.assertTrue(summary["complete"])
        self.assertEqual(verification["errors"], [])
        self.assertEqual(verification["execution_recovery"], note)
        self.assertEqual(verification["execution_recovery_source_sha256"],
                         report.sha256(self.root / "experiments/qwen/recovery_note.json"))
        self.assertIsNone(verification["unit_evidence"]["baby/qwen_semantic/seed42"]["training_numerics"])
        self.assertIsNone(verification["unit_evidence"]["sports/qwen_preference/seed44"]["training_numerics"])
        self.assertEqual(verification["unit_evidence"]["clothing/qwen_preference/seed44"]
                         ["training_numerics"]["candidate_training_sdpa_backend"], "math")
        zh, en = report.render_report(summary), report.render_report(summary, english=True)
        for text in (zh, en):
            self.assertIn("92956.gaas", text)
            self.assertIn("93023.gaas", text)
            self.assertIn("93033.gaas", text)
            self.assertIn("bitwise", text)
            self.assertIn("automatic", text)
        self.assertIn("首次序列化恢复", zh)
        self.assertIn("数值恢复与 backend 限制", zh)
        self.assertIn("不能确定具体 fused kernel 是根因", zh)
        self.assertIn("All six Clothing LoRA units", en)
        self.assertIn("does not establish a specific fused-kernel root cause", en)
        self.assertIn("not a claim that test computation never repeated", en)

    def test_pending_numerical_recovery_or_either_production_gate_blocks_completion(self):
        self.complete_fixture()
        self.clothing_numerics()
        for change in ({"status": "in_progress"}, {"production_same_state_regression": "pending"},
                       {"production_prefix100_regression": "pending"}):
            with self.subTest(change=change):
                self.numerical_recovery_note(**change)
                summary = report.build_report(self.root, verify_cache_files=True)
                self.assertEqual(summary["raw_record_count"], 27)
                self.assertFalse(summary["complete"])
                self.assertFalse(summary["verification"]["local_cache_bytes_verified"])
                self.assertTrue(any(issue["check"] == "recovery-regressions"
                                    for issue in summary["verification"]["errors"]))
                self.assertIn("INCOMPLETE", report.render_report(summary, english=True))

    def test_clothing_lora_requires_all_numeric_metadata_after_recovery(self):
        self.complete_fixture()
        self.numerical_recovery_note()
        missing = report.build_report(self.root)
        self.assertFalse(missing["complete"])
        self.assertEqual(sum(issue["check"] == "training-numerics"
                             for issue in missing["verification"]["errors"]), 6)
        self.clothing_numerics()
        good = {"candidate_training_sdpa_backend": "math", "nonfinite_gradient_guard": True,
                "cached_encoding_sdpa_backend": "automatic"}
        bad_values = [None, {}, {**good, "candidate_training_sdpa_backend": "automatic"},
                      {**good, "nonfinite_gradient_guard": False},
                      {**good, "nonfinite_gradient_guard": 1},
                      {**good, "cached_encoding_sdpa_backend": "math"}]
        for numerics in bad_values:
            with self.subTest(numerics=numerics):
                self.change_training("clothing", "qwen_preference", 44, training_numerics=numerics)
                summary = report.build_report(self.root)
                self.assertEqual(summary["raw_record_count"], 27)
                self.assertFalse(summary["complete"])
                self.assertTrue(any(issue["check"] == "training-numerics"
                                    and issue["unit"] == "clothing/qwen_preference/seed44"
                                    for issue in summary["verification"]["errors"]))

    def test_passed_numerical_recovery_does_not_waive_gradient_requirements(self):
        self.complete_fixture()
        self.clothing_numerics()
        self.numerical_recovery_note()
        self.change_training("clothing", "qwen_semantic", 42, gradient_evidence={})
        summary = report.build_report(self.root)
        self.assertFalse(summary["complete"])
        self.assertTrue(any(issue["check"] == "gradients" for issue in summary["verification"]["errors"]))

    def test_no_recovery_file_produces_no_recovery_section(self):
        summary = report.build_report(self.root)
        self.assertNotIn("execution_recovery", summary["verification"])
        self.assertNotIn("Actual execution recovery", report.render_report(summary, english=True))


if __name__ == "__main__":
    unittest.main()
