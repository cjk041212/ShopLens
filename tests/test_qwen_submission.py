"""Submission fixtures contain no real experiment, model, photo, or credentials."""
import importlib.util
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/build_qwen_submission.py"
SPEC = importlib.util.spec_from_file_location("build_qwen_submission", SCRIPT)
pack = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pack)


def put_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class SubmissionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def put(self, name, value="synthetic fixture"):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value, encoding="utf-8")
        return path

    def complete_fixture(self):
        for name in ("README.md", "README_QWEN.md", "requirements.txt", "requirements-qwen.txt", "pyproject.toml",
                     "experiments/qwen/protocol.md", "deliverables/ShopLens_Qwen_report.md",
                     "deliverables/ShopLens_Qwen_report_en.md", "deliverables/results_qwen_summary.csv",
                     "deliverables/ShopLens_Qwen_human_narration.md"):
            self.put(name)
        self.put("src/shoplens/example.py", '"""Synthetic code fixture."""\nvalue = 1\n')
        self.put("tests/test_example.py", "def test_fixture():\n    assert True\n")
        with zipfile.ZipFile(self.root / "deliverables/ShopLens_Qwen_presentation.pptx", "w") as pptx:
            pptx.writestr("ppt/slides/slide1.xml", "<slide>synthetic fixture, no measured results</slide>")
        put_json(self.root / "experiments/qwen/model_source.json", {"model_id": "fixture model"})
        put_json(self.root / "experiments/qwen/environment.json", {"user": "private-user", "torch": "fixture-version",
                 "cwd": "/home/private-user/nlp_coursework_shoplens", "cluster_host": "172.20.1.1"})
        proof = {"complete": True, "expected_records": 27, "summarized_records": 27, "errors": [], "missing_units": [],
                 "local_cache_bytes_verified": True, "unit_evidence": {}}
        for domain in pack.DOMAINS:
            old = [{"model": mode, "seed": seed, "split": "test"} for mode in pack.MODELS for seed in pack.SEEDS]
            put_json(self.root / "artifacts" / domain / "results.json", {"results": old})
            for seed in pack.SEEDS:
                put_json(self.root / "artifacts/budget_check" / domain / f"seed{seed}/budget_check.json", {"fixture_only": True})
            rows = []
            for mode in pack.QWEN_MODES:
                for seed in pack.SEEDS:
                    directory = self.root / "artifacts/qwen" / domain / mode / f"seed{seed}"
                    row = {"model": mode, "seed": seed, "split": "test", "fixture_only": True}
                    put_json(directory / "results.json", {"results": [row]})
                    put_json(directory / "training.json", {"diagnostic": False, "fixture_only": True})
                    put_json(directory / "models" / f"{mode}.json", {"diagnostic": False, "fixture_only": True})
                    put_json(directory / "complete.json", {"selected_epoch": 2, "results_sha256": pack.digest((directory / "results.json").read_bytes())})
                    proof["unit_evidence"][f"{domain}/{mode}/seed{seed}"] = {"fixture_only": True,
                        "source_evidence_sha256": {name: pack.file_digest(directory / name) for name in
                            ("training.json", f"models/{mode}.json", "complete.json", "results.json")}}
                    rows.append(row)
            put_json(self.root / "artifacts/qwen" / domain / "results.json", {"results": rows})
        put_json(self.root / "deliverables/verification.json", {"status": "passed", "formal_records": 54, "budget_records": 9, "errors": []})
        put_json(self.root / "experiments/qwen/verification.json", proof)
        put_json(self.root / "deliverables/results_qwen_summary.json", {"complete": True, "raw_record_count": 27, "verification": proof})
        retained = self.put("artifacts/retained_baseline_fixture.npz", "synthetic preserved baseline bytes")
        before = {"schema_version": 1, "hash_algorithm": "sha256", "files": {
            retained.relative_to(self.root).as_posix(): {"sha256": pack.file_digest(retained), "bytes": retained.stat().st_size}}}
        before["snapshot_sha256"] = pack.digest(json.dumps(before, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode())
        put_json(self.root / "experiments/qwen/baseline_before.json", before)
        put_json(self.root / "experiments/qwen/baseline_verification.json", {
            "unchanged": True, "matched_files": 1, "snapshot_sha256": before["snapshot_sha256"]})

    def update_proof(self, proof):
        put_json(self.root / "experiments/qwen/verification.json", proof)
        path = self.root / "deliverables/results_qwen_summary.json"
        summary = json.loads(path.read_text()); summary["verification"] = proof; put_json(path, summary)

    def test_incomplete_matrix_blocks_every_final_write(self):
        entries, manifest = pack.build_entries(self.root)
        self.assertFalse(manifest["experimental_status"]["all_final_requirements_met"])
        output = self.root / "final.zip"
        with self.assertRaisesRegex(pack.PackagingError, "Final build requires"):
            pack.write_archive(entries, output, manifest)
        self.assertFalse(output.exists())

    def test_complete_evidence_builds_verified_small_zip_without_source_changes(self):
        self.complete_fixture()
        source = self.root / "experiments/qwen/environment.json"
        original = source.read_bytes()
        entries, manifest = pack.build_entries(self.root)
        self.assertTrue(manifest["experimental_status"]["all_final_requirements_met"])
        output = self.root / "submission.zip"
        pack.write_archive(entries, output, manifest)
        self.assertEqual(source.read_bytes(), original)
        with zipfile.ZipFile(output) as archive:
            self.assertIsNone(archive.testzip())
            self.assertEqual(archive.read("ShopLens/README_QWEN.md"), b"synthetic fixture")
            clean = archive.read("ShopLens/experiments/qwen/environment.json").decode()
            self.assertNotIn("private-user", clean)
            self.assertNotIn("172.20.1.1", clean)
        record = next(row for row in manifest["files"] if row["path"] == "experiments/qwen/environment.json")
        self.assertNotEqual(record["source_sha256"], record["archive_sha256"])
        self.assertTrue(output.with_suffix(".zip.manifest.json").is_file())

    def test_weights_raw_reviews_photos_logs_and_previous_packages_are_excluded(self):
        self.complete_fixture()
        for name in ("data/raw/reviews_Baby_5.json.gz", "data/images/baby/A.jpg", "data/processed/baby/events.jsonl",
                     "data/hf-cache/model.safetensors", ".venv/bin/activate", ".ssh/config",
                     "artifacts/qwen/baby/qwen_semantic/seed42/candidate.npz",
                     "artifacts/qwen/baby/qwen_semantic/seed42/adapter/model.safetensors",
                     "logs/run.log", "experiments/qwen/logs/private.json", "deliverables/old.zip",
                     "deliverables/old.zip.manifest.json", "materials/course.pdf"):
            self.put(name)
        entries, _ = pack.build_entries(self.root)
        names = list(entries)
        for forbidden in ("data/images/", "data/hf-cache/", ".venv/", ".ssh/", "/logs/", "materials/", "events.jsonl",
                          "candidate.npz", ".safetensors", "old.zip"):
            self.assertFalse(any(forbidden in name for name in names), forbidden)

    def test_diagnostic_original_python_source_is_included_but_weights_are_not(self):
        self.complete_fixture()
        directory = "experiments/qwen/diagnostics/93023.gaas/original_source/"
        source = self.put(directory + "oldsource.py", 'ROOT = "/home/example/nlp_coursework_shoplens"\n')
        original = source.read_bytes()
        self.put(directory + "samefolder.pt")
        self.put(directory + "raw.jsonl")
        self.put("experiments/qwen/diagnostics/93023.gaas/unrelated.py")
        entries, manifest = pack.build_entries(self.root)
        name = "ShopLens/" + directory
        self.assertIn(name + "oldsource.py", entries)
        self.assertNotIn(name + "samefolder.pt", entries)
        self.assertNotIn(name + "raw.jsonl", entries)
        self.assertNotIn("ShopLens/experiments/qwen/diagnostics/93023.gaas/unrelated.py", entries)
        self.assertNotIn("/home/", entries[name + "oldsource.py"].decode())
        self.assertEqual(source.read_bytes(), original)
        record = next(row for row in manifest["files"] if row["path"] == directory + "oldsource.py")
        self.assertNotEqual(record["source_sha256"], record["archive_sha256"])

    def test_missing_budget_or_qwen_hash_verification_blocks_final(self):
        self.complete_fixture()
        (self.root / "artifacts/budget_check/baby/seed42/budget_check.json").unlink()
        status = pack.public_status(self.root)
        self.assertFalse(status["all_final_requirements_met"])
        self.assertEqual(status["completed_budget_checks"], 8)
        path = self.root / "experiments/qwen/verification.json"
        proof = json.loads(path.read_text()); proof["local_cache_bytes_verified"] = False; put_json(path, proof)
        self.assertFalse(pack.public_status(self.root)["qwen_verification_passed"])

    def test_baseline_preservation_failure_blocks_final(self):
        self.complete_fixture()
        path = self.root / "experiments/qwen/baseline_verification.json"
        proof = json.loads(path.read_text()); proof["unchanged"] = False; put_json(path, proof)
        self.assertFalse(pack.public_status(self.root)["baseline_preservation_verified"])
        self.assertFalse(pack.public_status(self.root)["all_final_requirements_met"])

    def test_stale_baseline_proof_and_tampered_snapshot_cannot_pass(self):
        self.complete_fixture()
        self.assertTrue(pack.public_status(self.root)["baseline_preservation_verified"])
        retained = self.root / "artifacts/retained_baseline_fixture.npz"
        original = retained.read_bytes()
        retained.write_bytes(b"changed after old verification")
        self.assertFalse(pack.public_status(self.root)["baseline_preservation_verified"])
        retained.write_bytes(original)
        path = self.root / "experiments/qwen/baseline_before.json"
        before = json.loads(path.read_text()); next(iter(before["files"].values()))["bytes"] += 1; put_json(path, before)
        self.assertFalse(pack.public_status(self.root)["baseline_preservation_verified"])

    def test_selected_json_changed_after_verification_blocks_final(self):
        self.complete_fixture()
        path = self.root / "artifacts/qwen/baby/qwen_semantic/seed42/training.json"
        training = json.loads(path.read_text()); training["modified_after_verification"] = True; put_json(path, training)
        status = pack.public_status(self.root)
        self.assertFalse(status["all_final_requirements_met"])
        self.assertTrue(any("Selected JSON evidence changed" in error for error in status["qwen_evidence_errors"]))

    def test_current_recovery_note_and_numerical_gates_are_checked_despite_complete_flags(self):
        self.complete_fixture()
        note = {"numerical_recovery": {"status": "in_progress", "production_same_state_regression": "passed",
                                       "production_prefix100_regression": "passed"}}
        path = self.root / "experiments/qwen/recovery_note.json"; put_json(path, note)
        status = pack.public_status(self.root)
        self.assertFalse(status["all_final_requirements_met"])
        self.assertTrue(any("Recovery note changed" in error for error in status["qwen_evidence_errors"]))
        proof = json.loads((self.root / "experiments/qwen/verification.json").read_text())
        proof.update(execution_recovery=note, execution_recovery_source_sha256=pack.file_digest(path))
        self.update_proof(proof)
        status = pack.public_status(self.root)
        self.assertFalse(status["all_final_requirements_met"])
        self.assertTrue(any("production regression gates" in error for error in status["qwen_evidence_errors"]))
        note["numerical_recovery"]["status"] = "complete"; put_json(path, note)
        proof.update(execution_recovery=note, execution_recovery_source_sha256=pack.file_digest(path))
        self.update_proof(proof)
        status = pack.public_status(self.root)
        self.assertEqual(sum("Missing Clothing MATH" in error for error in status["qwen_evidence_errors"]), 6)
        self.assertFalse(status["all_final_requirements_met"])
        for mode in ("qwen_semantic", "qwen_preference"):
            for seed in pack.SEEDS:
                directory = self.root / "artifacts/qwen/clothing" / mode / f"seed{seed}"
                training_path = directory / "training.json"
                training = json.loads(training_path.read_text())
                training["training_numerics"] = {"candidate_training_sdpa_backend": "math",
                    "nonfinite_gradient_guard": True, "cached_encoding_sdpa_backend": "automatic"}
                put_json(training_path, training)
                proof["unit_evidence"][f"clothing/{mode}/seed{seed}"]["source_evidence_sha256"]["training.json"] = pack.file_digest(training_path)
        self.update_proof(proof)
        self.assertTrue(pack.public_status(self.root)["all_final_requirements_met"])

    def test_unexpected_baseline_identity_cannot_count_as_complete(self):
        self.complete_fixture()
        path = self.root / "artifacts/baby/results.json"
        rows = json.loads(path.read_text()); rows["results"].append({"model": "diagnostic_extra", "seed": 42, "split": "test"})
        put_json(path, rows)
        status = pack.public_status(self.root)
        self.assertFalse(status["primary_complete"])
        self.assertFalse(status["all_final_requirements_met"])

    def test_existing_output_is_never_overwritten(self):
        self.complete_fixture()
        entries, manifest = pack.build_entries(self.root)
        output = self.put("existing.zip", "original archive fixture")
        with self.assertRaisesRegex(pack.PackagingError, "already exists"):
            pack.write_archive(entries, output, manifest)
        self.assertEqual(output.read_text(), "original archive fixture")

    def test_default_cli_is_dry_run_even_when_every_requirement_is_ready(self):
        self.complete_fixture()
        with redirect_stdout(io.StringIO()) as output:
            pack.main(["--root", str(self.root)])
        self.assertEqual(json.loads(output.getvalue())["mode"], "dry-run")
        self.assertFalse((self.root / "deliverables/ShopLens_Qwen_code_submission.zip").exists())

    def test_pbs_project_and_both_node_generator_defaults_are_portable_copies(self):
        self.complete_fixture()
        self.put("scripts/train_qwen.pbs", "#!/bin/bash\n#PBS -P private-project\necho hello\n")
        self.put("scripts/build_qwen_presentation.mjs", "const root = '/home/private-user/nlp_coursework_shoplens';\nconst runtime = '/home/private-user/runtime';\nconst skill = '/home/private-user/skill';\n")
        entries, manifest = pack.build_entries(self.root)
        self.assertNotIn("private-project", entries["ShopLens/scripts/train_qwen.pbs"].decode())
        script = entries["ShopLens/scripts/build_qwen_presentation.mjs"].decode()
        self.assertNotIn("/home/", script)
        self.assertIn("process.env.SHOPLENS_RUNTIME_BASE", script)
        self.assertIn("workspaceDir", script)
        self.assertIn("private-project", (self.root / "scripts/train_qwen.pbs").read_text())

    def test_jpg_and_jpeg_screenshots_are_included_as_unchanged_binary(self):
        self.complete_fixture()
        # Explicit binary signature fixture, not a real product photo or score.
        payload = b"\xff\xd8\xff\xe0JFIF\x00\x80\xfe\xff\xd9"
        for name in ("qwen_ui_desktop.jpg", "qwen_compare_baby.jpeg"):
            path = self.root / "deliverables/assets" / name
            path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(payload)
        entries, manifest = pack.build_entries(self.root)
        for name in ("qwen_ui_desktop.jpg", "qwen_compare_baby.jpeg"):
            relative = "deliverables/assets/" + name
            self.assertEqual(entries["ShopLens/" + relative], payload)
            self.assertEqual((self.root / relative).read_bytes(), payload)
            record = next(row for row in manifest["files"] if row["path"] == relative)
            self.assertEqual(record["source_sha256"], record["archive_sha256"])
        self.assertTrue(manifest["experimental_status"]["all_final_requirements_met"])

    def test_jpeg_extension_requires_jpeg_signature(self):
        with self.assertRaisesRegex(pack.PackagingError, "JPEG screenshot signature"):
            pack.sanitize(b"\x89PNG\r\n\x1a\n", Path("deliverables/assets/wrong.jpg"))

    def test_jpeg_private_path_metadata_is_rejected_without_changing_source(self):
        self.complete_fixture()
        path = self.root / "deliverables/assets/unsafe.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = b"\xff\xd8\xff\xe0metadata /home/example/private-file \xff\xd9"
        path.write_bytes(payload)
        with self.assertRaisesRegex(pack.PackagingError, "private absolute path"):
            pack.build_entries(self.root)
        self.assertEqual(path.read_bytes(), payload)


if __name__ == "__main__":
    unittest.main()
