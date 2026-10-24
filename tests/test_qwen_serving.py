"""Export/serving parity and exact temporal evaluation on a synthetic catalog."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from shoplens.models import TemporalExamples, digest, pad_histories, stable_topk
from shoplens.qwen_evaluate import evaluate_qwen
from shoplens.qwen_inference import QwenScorer, available_models, explain, load_scorer, recommend
from shoplens.qwen_models import QwenHistoryRecommender, export_serving


class QwenServingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.processed = self.root / "data" / "baby"
        self.unit = self.root / "artifacts" / "baby" / "qwen_preference" / "seed42"
        self.processed.mkdir(parents=True)
        (self.unit / "models").mkdir(parents=True)
        self.mode = "qwen_preference"
        self.ids = [f"item{i:02}" for i in range(32)]
        items = [{"item_id": item, "title": item, "text": "Example product text"} for item in self.ids]
        (self.processed / "items.jsonl").write_text("\n".join(map(json.dumps, items)))
        self.events = [
            {"user_id": "a", "item_id": self.ids[0], "timestamp": 1, "split": "train", "rating": 5},
            {"user_id": "b", "item_id": self.ids[1], "timestamp": 1, "split": "train", "rating": 5},
            {"user_id": "a", "item_id": self.ids[2], "timestamp": 2, "split": "test", "rating": 5},
            {"user_id": "a", "item_id": self.ids[3], "timestamp": 2, "split": "test", "rating": 5},
            {"user_id": "b", "item_id": self.ids[4], "timestamp": 2, "split": "test", "rating": 5},
            {"user_id": "a", "item_id": self.ids[0], "timestamp": 3, "split": "test", "rating": 5},
            {"user_id": "a", "item_id": self.ids[5], "timestamp": 3, "split": "test", "rating": 5}]
        (self.processed / "events.jsonl").write_text("\n".join(map(json.dumps, self.events)))
        torch.manual_seed(42)
        rng = np.random.default_rng(42)
        text, image, changed = (rng.normal(size=(32, dim)).astype(np.float32) for dim in (16, 8, 16))
        self.network = QwenHistoryRecommender(text, image, changed, embedding_dim=12, max_history=2).eval()
        self.head = self.unit / "models" / f"{self.mode}.pt"
        torch.save({"state_dict": self.network.state_dict()}, self.head)
        np.savez(self.unit / "serving.npz", item_ids=np.array(self.ids), **export_serving(self.network))
        self.meta = {"model": self.mode, "family": "qwen_candidate_lora", "seed": 42,
                     "item_ids": self.ids, "items_sha256": digest(self.processed / "items.jsonl"),
                     "events_sha256": digest(self.processed / "events.jsonl"), "embedding_dim": 12,
                     "max_history": 2, "epochs": 2, "selected_epoch": 1,
                     "training_item_counts": [0] * 32, "base_features_sha256": "fixture-base",
                     "candidate_sha256": "fixture-candidate", "head_sha256": digest(self.head),
                     "serving_sha256": digest(self.unit / "serving.npz")}
        self.save_meta()

    def save_meta(self):
        (self.unit / "models" / f"{self.mode}.json").write_text(json.dumps(self.meta))

    def test_export_matches_live_asymmetric_network(self):
        histories = [[0], [3, 2], [8, 7, 6, 5]]
        scorer = QwenScorer(self.processed, self.unit, self.mode)
        with torch.inference_mode():
            users = self.network.user_vectors(torch.from_numpy(pad_histories(histories, 2)))
            expected = (users @ self.network.item_vectors().T).numpy()
        np.testing.assert_allclose(scorer.score_histories(histories), expected, rtol=1e-5, atol=1e-6)
        self.assertFalse(torch.allclose(scorer.history_item_vectors, scorer.item_vectors))

    def test_partial_unit_not_advertised(self):
        self.assertEqual(available_models(self.unit), [])
        (self.unit / "complete.json").write_text('{}')
        self.assertEqual(available_models(self.unit), [self.mode])

    def test_catalog_and_serving_hashes_checked(self):
        self.meta["serving_sha256"] = "changed"
        self.save_meta()
        with self.assertRaisesRegex(ValueError, "serving_sha256"):
            QwenScorer(self.processed, self.unit, self.mode)

    def test_unknown_or_empty_history_rejected(self):
        for history in ([], ["unknown"]):
            with self.assertRaises(ValueError):
                recommend(self.processed, self.unit, self.mode, history)

    def test_ranking_excludes_all_likes_beyond_history_limit(self):
        liked = self.ids[:8]
        rows = recommend(self.processed, self.unit, self.mode, liked, k=32)
        self.assertEqual(len(rows), 24)
        self.assertFalse(set(liked) & {row["item_id"] for row in rows})
        evidence = explain(self.processed, self.unit, self.mode, liked, rows[0]["item_id"])
        self.assertEqual(len(evidence["evidence"]), 2)
        self.assertIn("not a causal", evidence["note"])

    def test_formal_metrics_match_explicit_user_macro(self):
        result = evaluate_qwen(self.processed, self.unit, self.mode, 42, device=torch.device("cpu"))
        self.assertEqual(result["run_protocol"]["device"], "cpu")
        scorer = load_scorer(self.processed, self.unit, self.mode)
        users = {}
        examples = TemporalExamples(self.events, self.ids, 2).examples("test")
        for row in examples:
            rank = stable_topk(scorer.score_histories([row["history"]])[0], 10, row["seen"])
            match = np.flatnonzero(rank == row["target"])
            users.setdefault(row["user_id"], []).append(0.0 if not len(match) else 1 / np.log2(match[0] + 2))
        expected = np.mean([np.mean(values) for values in users.values()])
        self.assertAlmostEqual(result["ndcg@10"], expected)
        self.assertEqual(result["n_targets"], 4)
        self.assertEqual(result["n_users"], 2)
        self.assertFalse(result["run_protocol"]["evaluation_subset"])
        self.assertTrue((self.unit.parent.parent / "results.json").is_file())

    def test_test_cannot_select_warmup_epoch(self):
        self.meta["selected_epoch"] = 0
        self.save_meta()
        with self.assertRaisesRegex(ValueError, "adaptation epoch"):
            evaluate_qwen(self.processed, self.unit, self.mode, 42, device="cpu", write=False)

    def test_seed_and_events_mismatch_rejected(self):
        with self.assertRaisesRegex(ValueError, "seed"):
            evaluate_qwen(self.processed, self.unit, self.mode, 43, device="cpu", write=False)
        self.meta["events_sha256"] = "changed"
        self.save_meta()
        with self.assertRaisesRegex(ValueError, "interactions"):
            evaluate_qwen(self.processed, self.unit, self.mode, 42, device="cpu", write=False)


if __name__ == "__main__":
    unittest.main()
