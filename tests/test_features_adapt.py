"""Small synthetic stubs verify the pipeline; they never download model weights."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch import nn

from shoplens import adapt as adaptation
from shoplens import features
from shoplens.data import write_json, write_jsonl


class NumberTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [int(value) for value in text.split()]

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(map(str, ids))


class TextStub:
    def __init__(self):
        self.tokenizer = NumberTokenizer()
        self.seen = []

    def encode(self, texts, **kwargs):
        self.seen.extend(texts)
        result = np.zeros((len(texts), 384), dtype=np.float32)
        for index, text in enumerate(texts):
            numbers = self.tokenizer.encode(text)
            result[index, 0] = sum(numbers) + 1
            result[index, 1] = len(numbers)
        return result / np.maximum(np.linalg.norm(result, axis=1, keepdims=True), 1e-12)


class Batch(dict):
    def to(self, device):
        return Batch({key: value.to(device) for key, value in self.items()})


class ProcessorStub:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()

    def __call__(self, images, return_tensors):
        return Batch(pixel_values=torch.tensor([picture.getpixel((0, 0))[0] for picture in images], dtype=torch.float32))


class ClipStub(nn.Module):
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()

    def get_image_features(self, pixel_values):
        result = torch.zeros((len(pixel_values), 512))
        result[:, 0] = pixel_values
        result[:, 1] = 1
        return result


class AutoStub(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Module()
        self.encoder.layer = nn.ModuleList([nn.Linear(8, 8) for _ in range(4)])
        self.embedding = nn.Linear(8, 8)


class TrainTextStub(nn.Module):
    def __init__(self):
        super().__init__()
        self.auto = AutoStub()
        self.pool_scale = nn.Parameter(torch.ones(8))
        self.saved = []
        self.encoded_batches = []
        self.encoded_modes = []
        self.received_metadata = []

    def __getitem__(self, index):
        assert index == 0
        return types.SimpleNamespace(auto_model=self.auto)

    def tokenize(self, texts):
        self.encoded_batches.append(list(texts))
        self.encoded_modes.append(self.training)
        vectors = []
        for text in texts:
            item = int(text.split()[0][4:])
            vectors.append([np.sin((item + 1) * (dimension + 1)) for dimension in range(8)])
        return {"x": torch.tensor(vectors, dtype=torch.float32),
                "task": "sentence-similarity", "modality": "text"}

    def forward(self, inputs):
        self.received_metadata.append((inputs["task"], inputs["modality"]))
        output = torch.tanh(self.auto.embedding(inputs["x"]))
        for layer in self.auto.encoder.layer:
            output = torch.tanh(layer(output))
        return {"sentence_embedding": output * self.pool_scale}

    def save(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        self.saved.append(path)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def catalog(self):
        processed = self.root / "processed"
        rows = []
        # Nonalphabetical order makes positional mixups visible.
        for item, color in [("B", 100), ("A", 10)]:
            path = self.root / f"{item}.png"
            Image.new("RGB", (4, 4), (color, 0, 0)).save(path)
            rows.append({"item_id": item, "text": str(color), "image_path": str(path)})
        write_jsonl(processed / "items.jsonl", rows)
        return processed

    def fake_feature_encoders(self):
        return (patch("sentence_transformers.SentenceTransformer", side_effect=lambda *a, **k: TextStub()),
                patch("transformers.CLIPModel", ClipStub), patch("transformers.CLIPProcessor", ProcessorStub))

    def test_long_text_all_tokens_are_chunked_and_aggregated(self):
        model = TextStub()
        output = features.encode_texts(model, [" ".join(map(str, range(701)))], batch_size=2)
        chunks = [model.tokenizer.encode(text) for text in model.seen]
        self.assertEqual([token for chunk in chunks for token in chunk], list(range(701)))
        self.assertEqual([len(chunk) for chunk in chunks], [224, 224, 224, 29])
        self.assertEqual(output.shape, (1, 384))
        np.testing.assert_allclose(np.linalg.norm(output, axis=1), [1], rtol=1e-6)

    def test_feature_rows_keep_id_order_and_provenance_hash(self):
        processed = self.catalog()
        artifact = self.root / "artifacts"
        a, b, c = self.fake_feature_encoders()
        with a, b, c, redirect_stdout(io.StringIO()):
            destination = features.encode(processed, artifact, self.root, device="cpu")
        with np.load(destination, allow_pickle=False) as saved:
            self.assertEqual(saved["item_ids"].tolist(), ["B", "A"])
            self.assertEqual(saved["image"].shape, (2, 512))
            self.assertEqual(saved["text"].shape, (2, 384))
            np.testing.assert_allclose(saved["image"][:, 1], 1 / np.sqrt(np.array([100, 10]) ** 2 + 1), rtol=1e-6)
        metadata = json.loads((artifact / "features.json").read_text())
        self.assertEqual(metadata["catalog_sha256"], features.sha256(processed / "items.jsonl"))
        self.assertEqual(metadata["features_sha256"], features.sha256(destination))

    def test_changed_catalog_is_not_silently_reused(self):
        processed = self.catalog()
        artifact = self.root / "artifacts"
        a, b, c = self.fake_feature_encoders()
        with a, b, c, redirect_stdout(io.StringIO()):
            features.encode(processed, artifact, self.root, device="cpu")
        rows = features.read_items(processed)
        rows[0]["text"] = "123"
        write_jsonl(processed / "items.jsonl", rows)
        with self.assertRaisesRegex(ValueError, "Catalog content changed"):
            features.encode(processed, artifact, self.root, device="cpu")

    def test_feature_cache_without_metadata_is_rejected(self):
        processed = self.catalog()
        artifact = self.root / "artifacts"
        artifact.mkdir()
        np.savez(artifact / "features.npz", item_ids=np.array(["B", "A"]),
                 image=np.ones((2, 512)), text=np.ones((2, 384)))
        with self.assertRaisesRegex(ValueError, "metadata|provenance|features.json"):
            features.encode(processed, artifact, self.root, device="cpu")

    def test_feature_cache_with_empty_metadata_is_rejected(self):
        processed = self.catalog()
        artifact = self.root / "artifacts"
        artifact.mkdir()
        np.savez(artifact / "features.npz", item_ids=np.array(["B", "A"]),
                 image=np.ones((2, 512)), text=np.ones((2, 384)))
        write_json(artifact / "features.json", {})
        with self.assertRaisesRegex(ValueError, "metadata|provenance|checksum"):
            features.encode(processed, artifact, self.root, device="cpu")

    def test_modified_feature_cache_is_rejected_by_checksum(self):
        processed = self.catalog()
        artifact = self.root / "artifacts"
        a, b, c = self.fake_feature_encoders()
        with a, b, c, redirect_stdout(io.StringIO()):
            destination = features.encode(processed, artifact, self.root, device="cpu")
        with np.load(destination, allow_pickle=False) as old:
            arrays = {key: old[key].copy() for key in old.files}
        arrays["text"][0, 0] += .1
        np.savez_compressed(destination, **arrays)
        with self.assertRaisesRegex(ValueError, "checksum"):
            features.encode(processed, artifact, self.root, device="cpu")

    def test_wrong_cached_encoder_ids_are_rejected(self):
        processed = self.catalog()
        artifact = self.root / "artifacts"
        a, b, c = self.fake_feature_encoders()
        with a, b, c, redirect_stdout(io.StringIO()):
            features.encode(processed, artifact, self.root, device="cpu")
        metadata_path = artifact / "features.json"
        metadata = json.loads(metadata_path.read_text())
        for key in ("clip_model", "text_model"):
            with self.subTest(key=key):
                write_json(metadata_path, {**metadata, key: "wrong/encoder"})
                with self.assertRaisesRegex(ValueError, "model IDs"):
                    features.encode(processed, artifact, self.root, device="cpu")

    def adaptation_fixture(self):
        processed = self.root / "adapt-processed"
        items = [{"item_id": f"item{i}", "text": f"item{i} original product description words"} for i in range(30)]
        reviews = [{"item_id": f"item{i}", "user_id": f"user{j}", "rating": 5,
                    "text": f"item{i} user review with several words variant{j}", "timestamp": 10}
                   for i in range(30) for j in range(3)]
        write_jsonl(processed / "items.jsonl", items)
        write_jsonl(processed / "reviews_train.jsonl", reviews)
        write_json(processed / "audit.json", {"cutoffs": {"train_end": 100}})
        return processed

    def test_adaptation_rejects_future_reviews_before_model_loading(self):
        processed = self.adaptation_fixture()
        rows = list(map(json.loads, (processed / "reviews_train.jsonl").read_text().splitlines()))
        rows[-1]["timestamp"] = 101
        write_jsonl(processed / "reviews_train.jsonl", rows)
        with patch.object(adaptation, "SentenceTransformer", side_effect=AssertionError("No model download allowed")):
            with self.assertRaisesRegex(ValueError, "cutoff"):
                adaptation.adapt(processed, self.root / "output", device="cpu")

    def test_adaptation_trains_last_two_layers_and_deduplicates_asins(self):
        processed = self.adaptation_fixture()
        torch.manual_seed(123)
        model = TrainTextStub()
        previous = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
        with patch.object(adaptation, "SentenceTransformer", return_value=model), redirect_stdout(io.StringIO()):
            adaptation.adapt(processed, self.root / "output", device="cpu", epochs=1, batch_size=16, lr=.01)
        for name, parameter in model.named_parameters():
            expected = name.startswith("auto.encoder.layer.2.") or name.startswith("auto.encoder.layer.3.")
            self.assertEqual(parameter.requires_grad, expected, name)
            if not expected:
                torch.testing.assert_close(parameter, previous[name], rtol=0, atol=0)
        self.assertTrue(any(not torch.equal(parameter, previous[name])
                            for name, parameter in model.named_parameters() if parameter.requires_grad))
        self.assertTrue(model.saved)
        self.assertTrue(model.received_metadata)
        self.assertEqual(set(model.received_metadata), {("sentence-similarity", "text")})
        self.assertTrue(model.encoded_batches)
        for batch in model.encoded_batches:
            asins = [text.split()[0] for text in batch]
            self.assertEqual(len(asins), len(set(asins)), "Same ASIN must not become its own contrastive negative")
        train_asins = {text.split()[0] for mode, batch in zip(model.encoded_modes, model.encoded_batches)
                       if mode for text in batch}
        validation_asins = {text.split()[0] for mode, batch in zip(model.encoded_modes, model.encoded_batches)
                            if not mode for text in batch}
        self.assertTrue(train_asins)
        self.assertTrue(validation_asins)
        self.assertFalse(train_asins & validation_asins)

    def test_encode_preserves_non_tensor_tokenizer_metadata(self):
        model = TrainTextStub()
        embeddings = adaptation._encode(model, ["item1 example sentence", "item2 another sentence"], "cpu")
        self.assertEqual(embeddings.shape, (2, 8))
        self.assertTrue(torch.isfinite(embeddings).all())
        torch.testing.assert_close(embeddings.norm(dim=1), torch.ones(2), rtol=1e-5, atol=1e-6)
        self.assertEqual(model.received_metadata, [("sentence-similarity", "text")])
        embeddings.sum().backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))

    def test_adaptation_rejects_zero_epochs(self):
        processed = self.adaptation_fixture()
        with patch.object(adaptation, "SentenceTransformer", side_effect=AssertionError("Validate before model loading")):
            with self.assertRaisesRegex(ValueError, "epochs"):
                adaptation.adapt(processed, self.root / "output", device="cpu", epochs=0)


if __name__ == "__main__":
    unittest.main()
