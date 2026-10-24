"""Synthetic protocol fixtures for real exported-head inference, never results.

All files live in diagnostic temporary directories. Formal-shaped manifests only
exercise the consumer contract; fake adapter bytes are checksummed, never loaded.
No pretrained model, dataset, network, training or formal output is used.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from shoplens.models import pad_histories
from shoplens.qwen_models import QwenHistoryRecommender, export_serving
from shoplens.qwen_text import LORA_CONFIG


MODULE_PATH = Path(__file__).resolve().parents[1] / "src/shoplens/studio_models.py"
spec = importlib.util.spec_from_file_location("shoplens.studio_models_diagnostic_fixture", MODULE_PATH)
models = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(models)


class TrainedStudioModelTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="shoplens-trained-studio-diagnostic-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.processed = self.root / "data/processed/baby"
        self.processed.mkdir(parents=True)
        self.ids = [f"DIAGNOSTIC-{index:02d}" for index in range(26)]
        items = [{"item_id": item_id, "title": f"Synthetic protocol fixture {item_id}", "fixture_only": True} for item_id in self.ids]
        self.catalog = self.processed / "items.jsonl"
        self.catalog.write_text("\n".join(json.dumps(item) for item in items))
        self.text, self.image = np.eye(26, 1024, dtype=np.float32), np.eye(26, 512, dtype=np.float32)
        old = self.root / "artifacts/baby/features.npz"
        base = self.root / "artifacts/qwen/baby/base/features.npz"
        old.parent.mkdir(parents=True)
        base.parent.mkdir(parents=True)
        np.savez(old, item_ids=self.ids, image=self.image, image_present=np.ones(26, dtype=bool))
        np.savez(base, item_ids=self.ids, text=self.text, image=self.image)
        old.with_suffix(".json").write_text(json.dumps({"fixture_only": True, "features_sha256": self.sha(old), "catalog_sha256": self.sha(self.catalog), "clip_model": "openai/clip-vit-base-patch32", "image_preprocessing": "CLIPProcessor + EXIF transpose"}))
        self.base_meta = {"identity": {"items_sha256": self.sha(self.catalog), "old_features_sha256": self.sha(old), "model_id": models.MODEL_ID, "revision": models.REVISION, "diagnostic": False},
                          "model_id": models.MODEL_ID, "revision": models.REVISION, "features_sha256": self.sha(base),
                          "encoding_config": {"max_length": 256, "padding_side": "left", "pooling": "last valid token", "normalization": "L2", "truncation": True}, "fixture_only": True}
        base.with_suffix(".json").write_text(json.dumps(self.base_meta))
        self.units, self.heads = {}, {}
        for offset, mode in enumerate(models.MODES):
            self.make_unit(mode, offset)
        models._inspect.cache_clear()
        models._verified.cache_clear()
        models._candidate_catalog.cache_clear()

    @staticmethod
    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def make_unit(self, mode, offset):
        unit = self.root / "artifacts/qwen/baby" / mode / "seed42"
        (unit / "models").mkdir(parents=True)
        candidate = np.roll(self.text, offset, axis=1)
        with torch.random.fork_rng():
            torch.manual_seed(17 + offset)
            head = QwenHistoryRecommender(self.text, self.image, candidate_text=candidate).eval()
        torch.save({"state_dict": head.state_dict()}, unit / "models" / f"{mode}.pt")
        np.savez(unit / "candidate.npz", item_ids=self.ids, text=candidate)
        np.savez(unit / "serving.npz", item_ids=self.ids, **export_serving(head))
        (unit / "results.json").write_text(json.dumps({"fixture_only": True, "note": "Diagnostic schema only; not an actual test result."}))
        adapter_sha = None
        if mode != "qwen_frozen":
            adapter = unit / "adapter"
            adapter.mkdir()
            (adapter / "adapter_config.json").write_text(json.dumps({"fixture_only": True}))
            (adapter / "adapter_model.safetensors").write_bytes(b"DIAGNOSTIC HASH FIXTURE ONLY: never load these bytes as a model")
            (adapter / "README.md").write_text("DIAGNOSTIC adapter provenance fixture")
            adapter_sha = models._tree_hash(adapter)
        fingerprint = hashlib.sha256(f"DIAGNOSTIC CONFIG {mode}".encode()).hexdigest()
        metadata = {"model": mode, "seed": 42, "family": "qwen_candidate_lora", "objective": mode,
                    "model_id": models.MODEL_ID, "revision": models.REVISION, "item_ids": self.ids,
                    "items_sha256": self.sha(self.catalog), "embedding_dim": 128, "max_history": 20,
                    "selected_epoch": 1, "epochs": 2, "config_sha256": fingerprint,
                    "features_metadata": self.base_meta, "base_features_sha256": self.base_meta["features_sha256"],
                    "data_audit": {"synthetic": False, "fixture_only": True}, "diagnostic": False,
                    "lora_config": None if mode == "qwen_frozen" else LORA_CONFIG,
                    "head_sha256": self.sha(unit / "models" / f"{mode}.pt"),
                    "serving_sha256": self.sha(unit / "serving.npz"), "candidate_sha256": self.sha(unit / "candidate.npz"),
                    "adapter_sha256": adapter_sha, "fixture_only": True}
        complete = {"model": mode, "seed": 42, "fingerprint": fingerprint, "selected_epoch": 1,
                    **{key: metadata[key] for key in ("head_sha256", "serving_sha256", "candidate_sha256", "adapter_sha256")},
                    "results_sha256": self.sha(unit / "results.json"), "fixture_only": True}
        (unit / "models" / f"{mode}.json").write_text(json.dumps(metadata))
        (unit / "complete.json").write_text(json.dumps(complete))
        self.units[mode], self.heads[mode] = unit, head

    def resign(self, mode, key, path):
        unit = self.units[mode]
        metadata_path = unit / "models" / f"{mode}.json"
        complete_path = unit / "complete.json"
        metadata, complete = json.loads(metadata_path.read_text()), json.loads(complete_path.read_text())
        metadata[key] = complete[key] = self.sha(path)
        metadata_path.write_text(json.dumps(metadata))
        complete_path.write_text(json.dumps(complete))

    def expected(self, mode, ids):
        unique = list(dict.fromkeys(ids))
        indices = [self.ids.index(item_id) for item_id in reversed(unique)]
        history = torch.as_tensor(pad_histories([indices], 20))
        head = self.heads[mode]
        with torch.inference_mode():
            return (head.user_vectors(history) @ head.item_vectors().T)[0].numpy()

    def test_status_is_lightweight_compact_and_reports_missing_units(self):
        with patch.object(models, "_sha", side_effect=AssertionError("Status must not hash large files")), \
             patch.object(models, "load_scorer", side_effect=AssertionError("Status must not load a scorer")):
            status = models.model_status(self.root, "baby")
        self.assertEqual(list(status), list(models.MODES))
        self.assertTrue(all(value["ready"] for value in status.values()))
        self.assertTrue(all(len(value["artifact_signature"]) == 64 for value in status.values()))
        self.assertNotIn("item_ids", json.dumps(status))
        (self.units["qwen_semantic"] / "complete.json").unlink()
        self.assertFalse(models.model_status(self.root, "baby")["qwen_semantic"]["ready"])

    def test_all_three_exported_heads_match_training_geometry_and_catalog_order(self):
        liked = self.ids[2:5]
        scores = []
        for mode in models.MODES:
            with self.subTest(mode=mode):
                result = models.personalization(self.root, "baby", mode, liked)
                np.testing.assert_allclose(result["scores"], self.expected(mode, liked), atol=1e-6, rtol=0)
                self.assertEqual(result["scores"].shape, (26,))
                self.assertTrue(result["metadata"]["active"])
                self.assertEqual(result["metadata"]["history_item_ids"], list(reversed(liked)))
                self.assertEqual(result["metadata"]["positive_count"], 3)
                self.assertEqual(result["metadata"]["seed"], 42)
                scores.append(result["scores"])
        self.assertFalse(np.allclose(scores[0], scores[1]))

    def test_duplicate_history_and_latest_twenty_follow_existing_serving_order(self):
        liked = self.ids[:25] + [self.ids[0]]
        result = models.personalization(self.root, "baby", "qwen_preference", liked)
        np.testing.assert_allclose(result["scores"], self.expected("qwen_preference", liked), atol=1e-6, rtol=0)
        self.assertEqual(result["metadata"]["positive_count"], 25)
        self.assertEqual(result["metadata"]["history_count"], 20)
        self.assertEqual(result["metadata"]["history_item_ids"], self.ids[24:4:-1])
        self.assertTrue(result["metadata"]["history_truncated"])

    def test_empty_history_is_inactive_but_still_requires_valid_exports(self):
        with patch.object(models, "load_scorer", side_effect=AssertionError("No empty-history scorer call")):
            result = models.personalization(self.root, "baby", "qwen_semantic", [])
        self.assertIsNone(result["scores"])
        self.assertFalse(result["metadata"]["active"])
        self.assertEqual(result["metadata"]["history_count"], 0)
        self.assertIn("No empty-history scores are invented", result["metadata"]["score_note"])
        (self.units["qwen_semantic"] / "complete.json").unlink()
        with self.assertRaises(FileNotFoundError):
            models.personalization(self.root, "baby", "qwen_semantic", [])

    def test_wrong_model_seed_and_completed_epoch_cannot_use_cached_scores(self):
        unit = self.units["qwen_frozen"]
        path = unit / "models/qwen_frozen.json"
        original = path.read_text()
        for change in ({"seed": 43}, {"model": "qwen_preference"}, {"selected_epoch": 2}):
            metadata = json.loads(original)
            metadata.update(change)
            path.write_text(json.dumps(metadata))
            with self.subTest(change=change):
                self.assertFalse(models.model_status(self.root, "baby")["qwen_frozen"]["ready"])
                with self.assertRaises(ValueError):
                    models.personalization(self.root, "baby", "qwen_frozen", self.ids[:2])
        path.write_text(original)

    def test_candidate_tampering_changes_signature_and_invalidates_verified_cache(self):
        mode = "qwen_frozen"
        models.personalization(self.root, "baby", mode, self.ids[:2])
        before = models.model_status(self.root, "baby")[mode]["artifact_signature"]
        with (self.units[mode] / "candidate.npz").open("ab") as stream:
            stream.write(b"DIAGNOSTIC-TAMPER")
        after = models.model_status(self.root, "baby")[mode]["artifact_signature"]
        self.assertNotEqual(before, after)
        with self.assertRaisesRegex(ValueError, "candidate_sha256"):
            models.personalization(self.root, "baby", mode, self.ids[:2])

    def test_adapter_tree_change_invalidates_even_with_restored_mtime_and_size(self):
        mode = "qwen_semantic"
        models.personalization(self.root, "baby", mode, [])
        before = models.model_status(self.root, "baby")[mode]["artifact_signature"]
        path = self.units[mode] / "adapter/README.md"
        original_stat = path.stat()
        value = path.read_bytes()
        path.write_bytes(b"X" + value[1:])
        os.utime(path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
        after = models.model_status(self.root, "baby")[mode]["artifact_signature"]
        self.assertNotEqual(before, after)
        with self.assertRaisesRegex(ValueError, "adapter content checksum"):
            models.personalization(self.root, "baby", mode, [])

    def test_legitimate_head_update_returns_new_actual_scores(self):
        mode = "qwen_preference"
        before = models.personalization(self.root, "baby", mode, self.ids[:3])
        with torch.no_grad():
            self.heads[mode].user_network[0].bias.add_(.75)
        path = self.units[mode] / "models" / f"{mode}.pt"
        torch.save({"state_dict": self.heads[mode].state_dict()}, path)
        self.resign(mode, "head_sha256", path)
        after = models.personalization(self.root, "baby", mode, self.ids[:3])
        self.assertNotEqual(before["metadata"]["artifact_signature"], after["metadata"]["artifact_signature"])
        self.assertFalse(np.allclose(before["scores"], after["scores"]))
        np.testing.assert_allclose(after["scores"], self.expected(mode, self.ids[:3]), atol=1e-6, rtol=0)

    def test_successive_feedback_reuses_content_verification_without_skipping_inference(self):
        mode = "qwen_frozen"
        models.personalization(self.root, "baby", mode, self.ids[:2])
        with patch.object(models, "_sha", side_effect=AssertionError("Unchanged artifacts must reuse verification")):
            result = models.personalization(self.root, "baby", mode, self.ids[2:5])
        np.testing.assert_allclose(result["scores"], self.expected(mode, self.ids[2:5]), atol=1e-6, rtol=0)

    def test_resigned_bad_head_schema_and_reordered_candidates_are_rejected(self):
        mode = "qwen_frozen"
        head_path = self.units[mode] / "models" / f"{mode}.pt"
        saved = torch.load(head_path, weights_only=True)
        saved["state_dict"].pop("candidate_text_projection.weight")
        torch.save(saved, head_path)
        self.resign(mode, "head_sha256", head_path)
        with self.assertRaisesRegex(ValueError, "tensor schema"):
            models.personalization(self.root, "baby", mode, [])
        candidate_path = self.units["qwen_semantic"] / "candidate.npz"
        np.savez(candidate_path, item_ids=self.ids[::-1], text=self.text)
        self.resign("qwen_semantic", "candidate_sha256", candidate_path)
        with self.assertRaisesRegex(ValueError, "candidate cache catalog order"):
            models.personalization(self.root, "baby", "qwen_semantic", [])

    def test_candidate_catalog_returns_actual_selected_matrices_and_compact_provenance(self):
        for offset, mode in enumerate(models.MODES):
            with self.subTest(mode=mode):
                result = models.candidate_catalog(self.root, "baby", mode)
                np.testing.assert_array_equal(result["text"], np.roll(self.text, offset, axis=1))
                self.assertEqual(result["item_ids"], self.ids)
                self.assertEqual(result["text"].shape, (26, 1024))
                self.assertEqual(result["text"].dtype, np.float32)
                self.assertFalse(result["text"].flags.writeable)
                metadata = result["metadata"]
                self.assertEqual(metadata["model"], mode)
                self.assertEqual(metadata["selected_epoch"], 1)
                self.assertEqual(metadata["model_id"], models.MODEL_ID)
                self.assertEqual(metadata["revision"], models.REVISION)
                self.assertEqual(metadata["seed"], 42)
                self.assertIn("not a formal", metadata["scope"])
                self.assertNotIn(str(self.root), json.dumps(metadata))
                self.assertNotIn("item_ids", metadata)
                self.assertEqual(models.export_metadata(self.root, "baby", mode), metadata)
                expected_path = None if mode == "qwen_frozen" else (self.units[mode] / "adapter").resolve()
                self.assertEqual(models.adapter_path(self.root, "baby", mode), expected_path)

    def test_candidate_cache_is_immutable_but_caller_copies_and_metadata_are_independent(self):
        result = models.candidate_catalog(self.root, "baby", "qwen_semantic")
        with self.assertRaises(ValueError):
            result["text"][0, 0] = 9
        with self.assertRaises(ValueError):
            result["text"].setflags(write=True)
        copied = result["text"].copy()
        copied[0, 0] = 9
        result["metadata"]["model"] = "CALLER MUTATION"
        result["item_ids"][0] = "CALLER MUTATION"
        with patch.object(models, "_sha", side_effect=AssertionError("Unchanged export must reuse validation")), \
             patch.object(models.np, "load", side_effect=AssertionError("Unchanged candidate must reuse matrix")):
            again = models.candidate_catalog(self.root, "baby", "qwen_semantic")
        np.testing.assert_array_equal(again["text"], np.roll(self.text, 1, axis=1))
        self.assertEqual(again["metadata"]["model"], "qwen_semantic")
        self.assertEqual(again["item_ids"], self.ids)

    def test_resigned_candidate_update_invalidates_matrix_cache(self):
        mode = "qwen_semantic"
        before = models.candidate_catalog(self.root, "baby", mode)
        path = self.units[mode] / "candidate.npz"
        changed = np.roll(self.text, 5, axis=1)
        np.savez(path, item_ids=self.ids, text=changed)
        self.resign(mode, "candidate_sha256", path)
        after = models.candidate_catalog(self.root, "baby", mode)
        self.assertNotEqual(before["metadata"]["artifact_signature"], after["metadata"]["artifact_signature"])
        np.testing.assert_array_equal(after["text"], changed)
        np.testing.assert_array_equal(before["text"], np.roll(self.text, 1, axis=1))

    def test_candidate_metadata_and_adapter_path_never_fall_back_on_corruption(self):
        mode = "qwen_preference"
        models.candidate_catalog(self.root, "baby", mode)
        with (self.units[mode] / "candidate.npz").open("ab") as stream:
            stream.write(b"DIAGNOSTIC TAMPER")
        for consumer in (models.candidate_catalog, models.export_metadata, models.adapter_path):
            with self.subTest(consumer=consumer.__name__), self.assertRaisesRegex(ValueError, "candidate_sha256"):
                consumer(self.root, "baby", mode)
        (self.units["qwen_semantic"] / "adapter/adapter_model.safetensors").unlink()
        with self.assertRaises(FileNotFoundError):
            models.candidate_catalog(self.root, "baby", "qwen_semantic")

    def test_changed_canonical_catalog_and_unknown_history_ids_never_fall_back(self):
        with self.assertRaisesRegex(ValueError, "Unknown liked item"):
            models.personalization(self.root, "baby", "qwen_frozen", ["UNKNOWN-DIAGNOSTIC-ID"])
        models.personalization(self.root, "baby", "qwen_frozen", self.ids[:2])
        self.catalog.write_text(self.catalog.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "Catalog SHA256"):
            models.personalization(self.root, "baby", "qwen_frozen", self.ids[:2])


if __name__ == "__main__":
    unittest.main()
