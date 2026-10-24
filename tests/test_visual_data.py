"""Synthetic export fixtures only; these checks never evaluate formal experiments."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
import torch
from torch import nn

from shoplens.models import stable_topk
from shoplens.qwen_inference import QwenScorer

MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "shoplens" / "visual_data.py"
spec = importlib.util.spec_from_file_location("shoplens.visual_data", MODULE_PATH)
visual = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(visual)


class RecordingScorer:
    """Calls the production masked history + nonlinear user-network score path."""

    def __init__(self, max_history=3):
        actual = QwenScorer.__new__(QwenScorer)
        actual.device = torch.device("cpu")
        actual.mode = "qwen_frozen"
        actual.max_history = max_history
        actual.item_ids = [f"FIXTURE-{index:02d}" for index in range(9)]
        actual.index = {item_id: index for index, item_id in enumerate(actual.item_ids)}
        actual.items = [{"item_id": item_id, "title": f"Synthetic item {index}"}
                        for index, item_id in enumerate(actual.item_ids)]
        values = torch.tensor([[1., .2, .1, .3], [.2, 1., .2, .1], [.1, .2, 1., .3],
                               [.2, .1, .3, 1.], [1., .8, .4, .2], [.3, 1., .8, .2],
                               [.2, .3, 1., .8], [.8, .2, .3, 1.], [.8, .7, .6, .5]])
        actual.item_vectors = torch.nn.functional.normalize(values, dim=-1)
        actual.history_item_vectors = torch.nn.functional.normalize(values.roll(1, 1), dim=-1)
        actual.user_network = nn.Sequential(nn.Linear(4, 4), nn.GELU(), nn.Linear(4, 4))
        with torch.no_grad():
            actual.user_network[0].weight.copy_(torch.tensor([[.8,.1,0.,.2],[.2,.9,.1,0.],[0.,.1,.7,.2],[.1,0.,.3,.8]]))
            actual.user_network[0].bias.copy_(torch.tensor([.1,-.1,.2,.05]))
            actual.user_network[2].weight.copy_(torch.tensor([[.9,.2,0.,0.],[0.,.7,.3,0.],[.2,0.,.8,.2],[0.,.1,.2,.8]]))
            actual.user_network[2].bias.copy_(torch.tensor([.05,.1,-.1,.1]))
        actual.user_network.eval()
        self.actual, self.calls = actual, []
        self.__dict__.update({key: value for key, value in actual.__dict__.items()})

    def score_histories(self, histories):
        self.calls.append([list(history) for history in histories])
        return self.actual.score_histories(histories)


class VisualDataTests(unittest.TestCase):
    def setUp(self):
        self.reference = RecordingScorer()
        self.selected = RecordingScorer()
        self.ids = self.reference.item_ids

    def test_projection_and_background_are_deterministic_and_do_not_score(self):
        first = visual.catalog_projection(self.reference, fit_size=5, seed=42)
        second = visual.catalog_projection(self.reference, fit_size=5, seed=42)
        np.testing.assert_array_equal(first["coordinates"], second["coordinates"])
        self.assertEqual(first["fit_item_ids"], second["fit_item_ids"])
        self.assertEqual(len(first["fit_item_ids"]), 5)
        self.assertGreaterEqual(sum(first["explained_variance_ratio"]), 0)
        self.assertLessEqual(sum(first["explained_variance_ratio"]), 1 + 1e-12)
        self.assertEqual(self.reference.calls, [])
        one = visual.build_item_map(self.reference, self.selected, self.ids[:2], sample_size=2, k=2, projection=first)
        two = visual.build_item_map(self.reference, self.selected, self.ids[:2], sample_size=2, k=2, projection=first)
        pd.testing.assert_frame_equal(one["points"], two["points"])

    def test_fixed_reference_coordinates_do_not_move_with_history_or_selected_tower(self):
        projection = visual.catalog_projection(self.reference, fit_size=6)
        first = visual.build_item_map(self.reference, self.selected, self.ids[:2], sample_size=9, projection=projection)
        self.selected.item_vectors = self.selected.item_vectors.flip(1)
        self.selected.actual.item_vectors = self.selected.item_vectors
        self.selected.mode = "qwen_preference"
        second = visual.build_item_map(self.reference, self.selected, self.ids[2:4], sample_size=9, projection=projection)
        pd.testing.assert_frame_equal(first["points"][["item_id", "x", "y"]], second["points"][["item_id", "x", "y"]])

    def test_map_explicitly_adds_every_like_and_actual_topk_and_excludes_all_selected(self):
        liked = self.ids[:5]
        result = visual.build_item_map(self.reference, self.selected, liked, sample_size=0, k=2)
        points = result["points"].set_index("item_id")
        self.assertTrue(set(liked).issubset(points.index))
        self.assertEqual(set(points.loc[liked, "role"]), {"history"})
        self.assertEqual(points.loc[liked, "active_history"].sum(), 3)
        self.assertEqual(result["active_history_ids"], list(reversed(liked))[:3])
        self.assertTrue(set(result["top_items"]["item_id"]).issubset(points.index))
        self.assertTrue(set(liked).isdisjoint(result["top_items"]["item_id"]))
        expected = stable_topk(self.selected.score_histories([[4, 3, 2]])[0], 2, range(5))
        self.assertEqual(result["top_items"]["item_id"].tolist(), [self.ids[i] for i in expected])
        self.assertEqual(len(points), len(liked) + 2)

    def test_map_without_likes_shows_only_background_and_no_invented_scores(self):
        result = visual.build_item_map(self.reference, self.selected, [], sample_size=4, k=2)
        self.assertEqual(len(result["points"]), 4)
        self.assertEqual(set(result["points"]["role"]), {"background"})
        self.assertTrue(result["points"]["score"].isna().all())
        self.assertTrue(result["points"]["mean_history_cosine"].isna().all())
        self.assertTrue(result["points"]["rank"].isna().all())
        self.assertTrue(result["top_items"].empty)
        self.assertEqual(result["active_history_ids"], [])
        self.assertEqual(self.selected.calls, [])

    def test_full_dimensional_cosines_normalize_each_tower_not_pca(self):
        self.selected.item_vectors *= 3
        self.selected.history_item_vectors *= 7
        result = visual.history_candidate_similarities(self.selected, self.ids[:3], self.ids[5:8])
        history = self.selected.history_item_vectors[[2, 1, 0]].numpy()
        candidates = self.selected.item_vectors[[5, 6, 7]].numpy()
        expected = history / np.linalg.norm(history, axis=1, keepdims=True) @ (candidates / np.linalg.norm(candidates, axis=1, keepdims=True)).T
        np.testing.assert_allclose(result["matrix"].to_numpy(), expected, atol=1e-7)
        self.assertEqual(result["matrix"].index.tolist(), result["history"]["item_id"].tolist())
        self.assertEqual(result["matrix"].columns.tolist(), result["candidates"]["item_id"].tolist())
        self.assertTrue((np.abs(result["matrix"].to_numpy()) <= 1).all())

    def test_leave_one_out_uses_exact_scores_full_catalog_ranks_and_no_backfill(self):
        liked = self.ids[:5]
        result = visual.leave_one_out(self.selected, liked, k=2, target_item_ids=[self.ids[5], self.ids[8]])
        self.assertTrue(result["available"])
        self.assertEqual(self.selected.calls, [[[4, 3, 2], [3, 2], [4, 2], [4, 3]]])
        self.assertEqual(result["eligible_catalog_size"], 4)
        self.assertEqual(result["summary"]["removed_item_id"].tolist(), [self.ids[4], self.ids[3], self.ids[2]])
        self.assertTrue((result["summary"]["remaining_history_size"] == 2).all())
        manual = self.selected.actual.score_histories([[4, 3, 2], [3, 2]])
        expected_rankings = [stable_topk(row, 9, range(5)) for row in manual]
        expected_ranks = [{self.ids[i]: rank for rank, i in enumerate(row, 1)} for row in expected_rankings]
        for row in result["effects"].query("removed_item_id == @self.ids[4]").to_dict("records"):
            target = self.selected.index[row["target_item_id"]]
            self.assertAlmostEqual(row["baseline_score"], float(manual[0, target]))
            self.assertAlmostEqual(row["without_score"], float(manual[1, target]))
            self.assertAlmostEqual(row["score_delta"], float(manual[1, target] - manual[0, target]))
            self.assertEqual(row["baseline_rank"], expected_ranks[0][row["target_item_id"]])
            self.assertEqual(row["without_rank"], expected_ranks[1][row["target_item_id"]])
        expected_change = np.abs(manual[1, 5:] - manual[0, 5:]).mean()
        self.assertAlmostEqual(result["summary"].iloc[0]["mean_abs_score_change"], float(expected_change), places=7)
        self.assertTrue(set(liked).isdisjoint(result["baseline"]["item_id"]))

    def test_one_history_item_reports_unavailable_without_empty_history_call(self):
        result = visual.leave_one_out(self.selected, [self.ids[0]], k=2)
        self.assertFalse(result["available"])
        self.assertIn("empty history", result["reason"])
        self.assertTrue(result["summary"].empty)
        self.assertTrue(result["effects"].empty)
        self.assertEqual(self.selected.calls, [[[0]]])
        self.assertEqual(len(result["baseline"]), 2)

    def test_empty_target_list_produces_no_fabricated_effects(self):
        result = visual.leave_one_out(self.selected, self.ids[:2], target_item_ids=[])
        self.assertTrue(result["available"])
        self.assertEqual(len(result["summary"]), 2)
        self.assertTrue(result["effects"].empty)

    def test_selected_target_has_no_eligible_rank_even_when_its_history_is_removed(self):
        result = visual.leave_one_out(self.selected, self.ids[:2], target_item_ids=[self.ids[0]])
        self.assertTrue(result["effects"]["baseline_rank"].isna().all())
        self.assertTrue(result["effects"]["without_rank"].isna().all())
        self.assertFalse(result["effects"]["eligible"].any())

    def test_unknown_items_empty_history_mismatched_catalog_and_zero_vectors_rejected(self):
        with self.assertRaises(ValueError):
            visual.build_item_map(self.reference, self.selected, ["unknown"])
        with self.assertRaises(ValueError):
            visual.leave_one_out(self.selected, [])
        with self.assertRaises(ValueError):
            visual.history_candidate_similarities(self.selected, self.ids[:1], ["unknown"])
        self.selected.item_ids = list(reversed(self.selected.item_ids))
        with self.assertRaises(ValueError):
            visual.build_item_map(self.reference, self.selected, self.ids[:1])
        self.reference.item_vectors[0] = 0
        with self.assertRaises(ValueError):
            visual.catalog_projection(self.reference)


if __name__ == "__main__":
    unittest.main()
