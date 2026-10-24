"""Tiny diagnostic encoder fixtures; no pretrained downloads or formal outputs."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image


MODULE_PATH = Path(__file__).resolve().parents[1] / "src/shoplens/studio_encoders.py"
spec = importlib.util.spec_from_file_location("shoplens.studio_encoders_diagnostic_fixture", MODULE_PATH)
encoders = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(encoders)


class StudioCatalogTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="shoplens-studio-diagnostic-fixture-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.catalog = self.root / "data/processed/baby/items.jsonl"
        self.base = self.root / "artifacts/qwen/baby/base/features.npz"
        self.old = self.root / "artifacts/baby/features.npz"
        for file in (self.catalog, self.base, self.old):
            file.parent.mkdir(parents=True, exist_ok=True)
        self.ids = ["DIAGNOSTIC-A", "DIAGNOSTIC-B"]
        items = [{"item_id": item_id, "title": f"Synthetic fixture {item_id}", "price": 12.5, "fixture_only": True} for item_id in self.ids]
        self.catalog.write_text("\n".join(json.dumps(item) for item in items))
        self.text, self.image = np.eye(2, 1024, dtype=np.float32), np.eye(2, 512, dtype=np.float32)
        np.savez(self.old, item_ids=self.ids, image=self.image, image_present=np.ones(2, dtype=bool))
        np.savez(self.base, item_ids=self.ids, text=self.text, image=self.image)
        self.old_meta = {"catalog_sha256": self.sha(self.catalog), "features_sha256": self.sha(self.old),
                         "clip_model": encoders.CLIP_ID, "image_preprocessing": "CLIPProcessor + EXIF transpose", "fixture_only": True}
        self.base_meta = {"identity": {"items_sha256": self.sha(self.catalog), "old_features_sha256": self.sha(self.old),
                                       "model_id": encoders.QWEN_ID, "revision": encoders.QWEN_REVISION, "diagnostic": False},
                          "model_id": encoders.QWEN_ID, "revision": encoders.QWEN_REVISION,
                          "encoding_config": {"max_length": 256, "padding_side": "left", "pooling": "last valid token", "normalization": "L2", "truncation": True},
                          "features_sha256": self.sha(self.base), "fixture_only": True}
        self.write_metadata()
        encoders._catalog_cached.cache_clear()

    @staticmethod
    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def write_metadata(self):
        self.base.with_suffix(".json").write_text(json.dumps(self.base_meta))
        self.old.with_suffix(".json").write_text(json.dumps(self.old_meta))

    def test_catalog_preserves_fields_and_returns_aligned_readonly_towers(self):
        result = encoders.load_catalog(self.root, "baby")
        self.assertEqual(result["item_ids"], self.ids)
        self.assertEqual(result["items"][0]["price"], 12.5)
        np.testing.assert_array_equal(result["text"], self.text)
        np.testing.assert_array_equal(result["image"], self.image)
        self.assertFalse(result["text"].flags.writeable)
        self.assertFalse(result["image"].flags.writeable)
        self.assertEqual(result["signature"], ("baby", self.sha(self.catalog), self.sha(self.base), self.sha(self.old)))
        self.assertIn("not a CLIP revision", result["metadata"]["clip_revision_note"])

    def test_catalog_and_npz_tampering_fail_checksum_validation(self):
        self.catalog.write_text(self.catalog.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "Catalog SHA256"):
            encoders.load_catalog(self.root, "baby")
        self.base_meta["identity"]["items_sha256"] = self.sha(self.catalog)
        self.old_meta["catalog_sha256"] = self.sha(self.catalog)
        self.write_metadata()
        with self.base.open("ab") as output:
            output.write(b"DIAGNOSTIC-TAMPER")
        with self.assertRaisesRegex(ValueError, "base feature cache SHA256"):
            encoders.load_catalog(self.root, "baby")

    def test_reordered_items_and_changed_clip_vectors_are_rejected(self):
        np.savez(self.base, item_ids=self.ids[::-1], text=self.text, image=self.image)
        self.base_meta["features_sha256"] = self.sha(self.base)
        self.write_metadata()
        with self.assertRaisesRegex(ValueError, "item order"):
            encoders.load_catalog(self.root, "baby")
        np.savez(self.base, item_ids=self.ids, text=self.text, image=self.image[::-1])
        self.base_meta["features_sha256"] = self.sha(self.base)
        self.write_metadata()
        with self.assertRaisesRegex(ValueError, "original aligned CLIP"):
            encoders.load_catalog(self.root, "baby")

    def test_diagnostic_provenance_and_wrong_encoder_protocol_are_rejected(self):
        self.base_meta["diagnostic"] = True
        self.write_metadata()
        with self.assertRaisesRegex(ValueError, "non-diagnostic"):
            encoders.load_catalog(self.root, "baby")
        self.base_meta["diagnostic"] = False
        self.base_meta["encoding_config"]["padding_side"] = "right"
        self.write_metadata()
        with self.assertRaisesRegex(ValueError, "encoding protocol"):
            encoders.load_catalog(self.root, "baby")


class StudioEncoderTests(unittest.TestCase):
    def setUp(self):
        encoders._query_cached.cache_clear()
        encoders._query_runtime.cache_clear()
        encoders._adapted_query_cached.cache_clear()
        encoders._image_cached.cache_clear()

    @staticmethod
    def png(size=(3, 2)):
        output = io.BytesIO()
        Image.new("RGB", size, "red").save(output, format="PNG")
        return output.getvalue()

    def test_status_is_lazy_and_missing_qwen_has_actionable_error(self):
        with patch.object(encoders, "_snapshot", return_value=None), patch.object(encoders, "_query_model") as loader:
            result = encoders.status("/DIAGNOSTIC-NONEXISTENT")
            self.assertFalse(result["ready"])
            self.assertFalse(result["query_ready"])
            self.assertFalse(result["image_ready"])
            self.assertIn("prepare_studio_models.py", result["qwen"]["error"])
            with self.assertRaisesRegex(RuntimeError, "not prepared locally"):
                encoders.encode_query("/DIAGNOSTIC-NONEXISTENT", "a synthetic shopping request")
            loader.assert_not_called()

    def test_query_calls_shared_protocol_with_instruction_and_caches_real_result(self):
        recorded = []
        def embed(model, tokenizer, texts, device, max_length):
            recorded.append((texts, device, max_length))
            return torch.arange(1, 1025, dtype=torch.float32)[None, :]
        with patch.object(encoders, "_model_path", return_value=("diagnostic fixture", "cpu", ())), \
             patch.object(encoders, "_query_model", return_value=(object(), object())), \
             patch("shoplens.qwen_text.embed_texts", side_effect=embed):
            vector = encoders.encode_query("/DIAGNOSTIC", "  red running shoes  ")
            again = encoders.encode_query("/DIAGNOSTIC", "red running shoes")
        self.assertEqual(vector.shape, (1024,))
        self.assertEqual(vector.dtype, np.float32)
        self.assertAlmostEqual(float(np.linalg.norm(vector)), 1, places=6)
        np.testing.assert_array_equal(vector, again)
        self.assertEqual(len(recorded), 1)
        self.assertEqual(recorded[0], ([f"Instruct: {encoders.QUERY_INSTRUCTION}\nQuery:red running shoes"], "cpu", 256))
        self.assertIsNot(vector, again)

    def test_image_uses_exif_rgb_and_returns_normalized_512_vector(self):
        output = io.BytesIO()
        picture = Image.new("RGB", (2, 3), "blue")
        exif = picture.getexif()
        exif[274] = 6
        picture.save(output, format="JPEG", exif=exif)
        observed = []
        class Batch(dict):
            def to(self, device):
                return self
        def processor(images, return_tensors):
            observed.append((images[0].mode, images[0].size, return_tensors))
            return Batch(pixel_values=torch.ones(1))
        class Model:
            def get_image_features(self, **inputs):
                return torch.arange(1, 513, dtype=torch.float32)[None, :]
        with patch.object(encoders, "_model_path", return_value=("diagnostic fixture", "cpu", ())), \
             patch.object(encoders, "_image_model", return_value=(Model(), processor)):
            vector = encoders.encode_image("/DIAGNOSTIC", output.getvalue())
            again = encoders.encode_image("/DIAGNOSTIC", output.getvalue())
        self.assertEqual(vector.shape, (512,))
        self.assertAlmostEqual(float(np.linalg.norm(vector)), 1, places=6)
        self.assertEqual(observed, [("RGB", (3, 2), "pt")])
        np.testing.assert_array_equal(vector, again)

    def test_unsafe_images_fail_before_loading_any_model(self):
        with patch.object(encoders, "_image_model") as loader:
            with self.assertRaisesRegex(ValueError, "invalid"):
                encoders.encode_image("/DIAGNOSTIC", b"not image data")
            with patch.object(encoders, "MAX_IMAGE_BYTES", 5), self.assertRaisesRegex(ValueError, "10 MB"):
                encoders.encode_image("/DIAGNOSTIC", self.png())
            with patch.object(encoders, "MAX_IMAGE_PIXELS", 4), self.assertRaisesRegex(ValueError, "megapixels"):
                encoders.encode_image("/DIAGNOSTIC", self.png())
            loader.assert_not_called()

    def test_bad_encoder_values_never_become_random_fallbacks(self):
        for vector in (np.zeros(1024), np.full(1024, np.nan), np.ones(10)):
            with self.subTest(shape=vector.shape), self.assertRaisesRegex(ValueError, "no fallback"):
                encoders._normalized(vector, 1024)


class AdaptedQueryEncoderTests(unittest.TestCase):
    """Real tiny Qwen3 + PEFT adapters; no downloaded weights or training."""

    def setUp(self):
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import Qwen3Config, Qwen3Model

        self.temporary = tempfile.TemporaryDirectory(prefix="shoplens-query-peft-diagnostic-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config = Qwen3Config(vocab_size=32, hidden_size=1024, intermediate_size=32, num_hidden_layers=1,
                                 num_attention_heads=8, num_key_value_heads=1, head_dim=128,
                                 pad_token_id=0, eos_token_id=2, bos_token_id=1, attention_dropout=0.)
        with torch.random.fork_rng():
            torch.manual_seed(23)
            self.base = Qwen3Model(self.config).eval().requires_grad_(False)
        self.base_state = {key: value.detach().clone() for key, value in self.base.state_dict().items()}
        self.modes = ("qwen_frozen", "qwen_semantic", "qwen_preference")
        self.paths, self.overrides, self.invalid = {}, {}, set()
        for offset, (domain, mode) in enumerate((d, m) for d in ("sports", "baby") for m in self.modes[1:]):
            with torch.random.fork_rng():
                model = Qwen3Model(self.config)
            model.load_state_dict(self.base_state)
            model = get_peft_model(model, LoraConfig(task_type=TaskType.FEATURE_EXTRACTION, r=2, lora_alpha=4,
                                                    target_modules=["q_proj", "v_proj"], lora_dropout=.05, bias="none"))
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    if "lora_" in name:
                        values = torch.arange(parameter.numel(), dtype=parameter.dtype).reshape(parameter.shape)
                        parameter.copy_(.03 * torch.sin(values * .17 + offset + (1 if "lora_B" in name else 0)))
            path = self.root / "DIAGNOSTIC_ONLY" / domain / mode
            model.save_pretrained(path, safe_serialization=True, save_embedding_layers=False)
            config_path = path / "adapter_config.json"
            config = json.loads(config_path.read_text())
            config["base_model_name_or_path"] = "/DIAGNOSTIC_OLD_CLUSTER_PATH_MUST_NOT_BE_LOADED"
            config_path.write_text(json.dumps(config))
            self.paths[domain, mode] = path
        self.tokenizer = self.Tokenizer()
        self.base_loads = 0
        self.api = SimpleNamespace(MODES=self.modes, export_metadata=self.export_metadata, adapter_path=self.adapter_path)
        for cached in (encoders._query_cached, encoders._adapted_query_cached, encoders._query_runtime):
            cached.cache_clear()
        self.patches = [patch.object(encoders, "_export_api", return_value=self.api),
                        patch.object(encoders, "_model_path", return_value=("DIAGNOSTIC_PINNED_BASE", "cpu", ("tiny",))),
                        patch.object(encoders, "_query_model", side_effect=self.load_base)]
        for context in self.patches:
            context.start()
        self.addCleanup(lambda: [context.stop() for context in reversed(self.patches)])
        self.addCleanup(encoders._query_runtime.cache_clear)

    class Tokenizer:
        padding_side = "left"
        def __call__(self, texts, padding=True, truncation=True, max_length=256, return_tensors="pt"):
            values = [[1] + [3 + int(hashlib.sha256(word.encode()).hexdigest()[:8], 16) % 29 for word in text.split()] + [2] for text in texts]
            values = [value[:max_length] for value in values]
            width = max(map(len, values))
            return {"input_ids": torch.tensor([[0] * (width - len(value)) + value for value in values]),
                    "attention_mask": torch.tensor([[0] * (width - len(value)) + [1] * len(value) for value in values])}

    def load_base(self, path, device, signature):
        self.assertEqual(path, "DIAGNOSTIC_PINNED_BASE")
        self.base_loads += 1
        return self.base, self.tokenizer

    def adapter_path(self, root, domain, mode):
        return None if mode == "qwen_frozen" else self.overrides.get((domain, mode), self.paths[domain, mode])

    def export_metadata(self, root, domain, mode):
        if mode in self.invalid:
            raise ValueError("DIAGNOSTIC selected export validation failed")
        path = self.adapter_path(root, domain, mode)
        digest = hashlib.sha256()
        if path is not None:
            for file in sorted(path.iterdir()):
                if file.is_file():
                    digest.update(file.name.encode())
                    digest.update(file.read_bytes())
        adapter_sha = digest.hexdigest() if path is not None else None
        return {"model": mode, "label": mode, "seed": 42, "selected_epoch": 1,
                "model_id": encoders.QWEN_ID, "revision": encoders.QWEN_REVISION,
                "adapter_sha256": adapter_sha,
                "artifact_signature": hashlib.sha256(f"{domain}|{mode}|{adapter_sha}".encode()).hexdigest(),
                "fixture_only": True}

    def reference(self, domain, mode, text):
        from peft import PeftModel
        from transformers import Qwen3Model
        from shoplens.qwen_text import embed_texts, review_query
        with torch.random.fork_rng():
            model = Qwen3Model(self.config)
        model.load_state_dict(self.base_state)
        if mode != "qwen_frozen":
            model = PeftModel.from_pretrained(model, self.adapter_path(self.root, domain, mode), is_trainable=False, local_files_only=True)
        model.eval().requires_grad_(False)
        with torch.inference_mode():
            return embed_texts(model, self.tokenizer, [review_query(text, encoders.QUERY_INSTRUCTION)], "cpu", 256)[0].numpy()

    def assert_disabled_and_frozen(self):
        from peft.tuners.tuners_utils import BaseTunerLayer
        runtime = encoders._query_runtime("DIAGNOSTIC_PINNED_BASE", "cpu", ("tiny",))
        self.assertTrue(all(layer.disable_adapters for layer in runtime["model"].modules() if isinstance(layer, BaseTunerLayer)))
        self.assertFalse(runtime["model"].training)
        self.assertTrue(all(not parameter.requires_grad for parameter in runtime["model"].parameters()))

    def test_real_adapters_match_each_encoder_and_frozen_survives_fresh_query(self):
        text = "A mat for home yoga"
        result = encoders.encode_queries(self.root, "sports", text)
        self.assertEqual(tuple(result), self.modes)
        for mode in self.modes:
            with self.subTest(mode=mode):
                np.testing.assert_allclose(result[mode], self.reference("sports", mode, text), atol=1e-6, rtol=0)
                self.assertEqual(result[mode].shape, (1024,))
                self.assertAlmostEqual(float(np.linalg.norm(result[mode])), 1, places=6)
        self.assertFalse(np.allclose(result["qwen_frozen"], result["qwen_semantic"]))
        self.assertFalse(np.allclose(result["qwen_semantic"], result["qwen_preference"]))
        fresh = "Another completely fresh shopping request"
        np.testing.assert_allclose(encoders.encode_query(self.root, fresh), self.reference("sports", "qwen_frozen", fresh), atol=1e-6, rtol=0)
        self.assertEqual(self.base_loads, 1)
        self.assert_disabled_and_frozen()

    def test_domain_and_changed_adapter_identity_do_not_reuse_wrong_vectors(self):
        text = "A mat for home yoga"
        sports = encoders.encode_queries(self.root, "sports", text)
        baby = encoders.encode_queries(self.root, "baby", text)
        np.testing.assert_array_equal(sports["qwen_frozen"], baby["qwen_frozen"])
        self.assertFalse(np.allclose(sports["qwen_semantic"], baby["qwen_semantic"]))
        before = encoders._adapted_query_cached.cache_info()
        encoders.encode_queries(self.root, "sports", text)
        self.assertEqual(encoders._adapted_query_cached.cache_info().hits, before.hits + 2)
        self.overrides["sports", "qwen_semantic"] = self.paths["baby", "qwen_semantic"]
        updated = encoders.encode_queries(self.root, "sports", text)
        np.testing.assert_allclose(updated["qwen_semantic"], baby["qwen_semantic"], atol=1e-6, rtol=0)
        self.assertFalse(np.allclose(updated["qwen_semantic"], sports["qwen_semantic"]))
        np.testing.assert_array_equal(updated["qwen_preference"], sports["qwen_preference"])
        self.assertEqual(self.base_loads, 1)
        self.assert_disabled_and_frozen()

    def test_metadata_preserves_export_signature_and_instruction_changes_cache_keys(self):
        text = "A mat for home yoga"
        before = encoders.query_encoding_metadata(self.root, "sports")
        self.assertEqual(len({details["query_protocol_hash"] for details in before.values()}), 1)
        encoders.encode_queries(self.root, "sports", text)
        for mode in self.modes:
            self.assertEqual(before[mode]["artifact_signature"], self.export_metadata(self.root, "sports", mode)["artifact_signature"])
            self.assertEqual(before[mode]["model_id"], encoders.QWEN_ID)
            self.assertIn("not the formal", before[mode]["scope"])
        frozen_misses = encoders._query_cached.cache_info().misses
        adapted_misses = encoders._adapted_query_cached.cache_info().misses
        with patch.object(encoders, "QUERY_INSTRUCTION", "A different diagnostic shopping instruction."):
            after = encoders.query_encoding_metadata(self.root, "sports")
            changed = encoders.encode_queries(self.root, "sports", text)
            np.testing.assert_allclose(changed["qwen_frozen"], self.reference("sports", "qwen_frozen", text), atol=1e-6, rtol=0)
        self.assertNotEqual(before["qwen_frozen"]["query_protocol_hash"], after["qwen_frozen"]["query_protocol_hash"])
        self.assertEqual(encoders._query_cached.cache_info().misses, frozen_misses + 1)
        self.assertEqual(encoders._adapted_query_cached.cache_info().misses, adapted_misses + 2)

    def test_export_or_forward_failure_raises_without_mode_fallback_and_restores_base(self):
        from peft import PeftModel
        from shoplens.qwen_text import embed_texts
        self.invalid.add("qwen_semantic")
        with self.assertRaisesRegex(ValueError, "export validation failed"):
            encoders.encode_queries(self.root, "sports", "A mat for home yoga")
        self.assertEqual(self.base_loads, 0)
        self.invalid.clear()
        def fail_adapter(model, *args, **kwargs):
            if isinstance(model, PeftModel):
                raise RuntimeError("DIAGNOSTIC adapter forward failed")
            return embed_texts(model, *args, **kwargs)
        with patch("shoplens.qwen_text.embed_texts", side_effect=fail_adapter), self.assertRaisesRegex(RuntimeError, "adapter forward failed"):
            encoders.encode_queries(self.root, "sports", "A mat for home yoga")
        self.assert_disabled_and_frozen()
        fresh = "A fresh request after the injected failure"
        np.testing.assert_allclose(encoders.encode_query(self.root, fresh), self.reference("sports", "qwen_frozen", fresh), atol=1e-6, rtol=0)

    def test_concurrent_queries_hold_one_lock_for_adapter_selection_and_forward(self):
        from shoplens.qwen_text import embed_texts
        class TrackingLock:
            def __init__(self):
                self.lock, self.owner = threading.RLock(), None
            def __enter__(self):
                self.lock.acquire()
                self.owner = threading.get_ident()
                return self
            def __exit__(self, *args):
                self.owner = None
                self.lock.release()
        lock = TrackingLock()
        observed = []
        def checked_forward(*args, **kwargs):
            self.assertEqual(lock.owner, threading.get_ident())
            observed.append(threading.get_ident())
            return embed_texts(*args, **kwargs)
        with patch.object(encoders, "_ENCODE_LOCK", lock), patch("shoplens.qwen_text.embed_texts", side_effect=checked_forward):
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(encoders.encode_queries, self.root, domain, "Concurrent diagnostic request " + domain) for domain in ("sports", "baby")]
                result = [future.result(timeout=20) for future in futures]
        self.assertEqual(len(set(observed)), 2)
        for index, domain in enumerate(("sports", "baby")):
            np.testing.assert_allclose(result[index]["qwen_semantic"], self.reference(domain, "qwen_semantic", "Concurrent diagnostic request " + domain), atol=1e-6, rtol=0)
        self.assertEqual(self.base_loads, 1)
        self.assert_disabled_and_frozen()


if __name__ == "__main__":
    unittest.main()
