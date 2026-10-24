"""Deterministic chart fixtures; these are not model or performance results."""

from __future__ import annotations

import importlib.util
import math
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "shoplens" / "visual_charts.py"
spec = importlib.util.spec_from_file_location("shoplens_visual_chart_fixture", MODULE_PATH)
charts = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(charts)


class RecommendationChartTests(unittest.TestCase):
    def setUp(self):
        self.catalog = {key: {"item_id": key, "title": f"Fixture product {key}"} for key in ("A", "B", "C")}
        self.results = {
            "qwen_frozen": [{"item_id": "A", "rank": 1, "score": 9000}, {"item_id": "B", "rank": 2, "score": 8000}],
            "qwen_semantic": [{"item_id": "B", "rank": 1, "score": -90}],
            "qwen_preference": [{"item_id": "A", "rank": 2, "score": .01}, {"item_id": "C", "rank": 1, "score": .02}],
        }

    def test_rank_flow_preserves_true_ranks_and_breaks_absent_paths(self):
        figure = charts.rank_flow_figure(self.results, self.catalog)
        self.assertEqual(list(figure.data[0].y), [1, None, 2])
        self.assertEqual(list(figure.data[1].y), [2, 1, None])
        self.assertEqual(list(figure.data[2].y), [None, None, 1])
        self.assertTrue(all(trace.connectgaps is False for trace in figure.data))
        self.assertEqual(list(figure.data[0].marker.color), ["#546982", "#347CDB", "#1A967B"])
        self.assertNotIn("score", figure.data[0].hovertemplate)
        self.assertIn("ranks outside Top-K are unknown", figure.layout.annotations[0].text)

    def test_duplicate_items_or_ranks_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "must not repeat"):
            charts.rank_flow_figure({"x": [{"item_id": "A", "rank": 1}, {"item_id": "A", "rank": 2}]}, self.catalog)
        with self.assertRaisesRegex(ValueError, "must not repeat"):
            charts.overlap_figure({"x": [{"item_id": "A", "rank": 1}, {"item_id": "B", "rank": 1}]})
        with self.assertRaisesRegex(ValueError, "positive integers"):
            charts.overlap_figure({"x": [{"item_id": "A", "rank": math.nan}]})

    def test_jaccard_uses_sets_not_ranks_or_scores(self):
        figure = charts.overlap_figure(self.results)
        z = figure.data[0].z
        self.assertEqual(z[0][0], 1)
        self.assertEqual(z[0][1], .5)
        self.assertAlmostEqual(z[0][2], 1 / 3)
        self.assertEqual(z[1][2], 0)
        self.assertEqual(list(figure.data[0].customdata[0][2]), [1, 3, 2, 2])

    def test_two_empty_sets_are_undefined_not_identical_recommendations(self):
        figure = charts.overlap_figure({"qwen_frozen": [], "qwen_semantic": [{"item_id": "A", "rank": 1}]})
        self.assertIsNone(figure.data[0].z[0][0])
        self.assertEqual(figure.data[0].z[0][1], 0)
        self.assertEqual(figure.data[0].text[0][0], "No items")
        self.assertEqual(len(charts.rank_flow_figure({}, {}).data), 0)


class EvidenceChartTests(unittest.TestCase):
    def setUp(self):
        self.history = [{"item_id": "H1", "title": "Actual history 1"}, {"item_id": "H2", "title": "Actual history 2"}]
        self.candidates = [{"item_id": "C1", "title": "Actual candidate 1"}, {"item_id": "C2", "title": "Actual candidate 2"}]
        self.values = [[-.4, .2], [.8, 0.0]]

    def test_heatmap_preserves_negative_values_and_zero_midpoint(self):
        figure = charts.evidence_heatmap(self.history, self.candidates, self.values)
        trace = figure.data[0]
        self.assertEqual([list(row) for row in trace.z], self.values)
        self.assertEqual((trace.zmin, trace.zmid, trace.zmax), (-1, 0, 1))
        self.assertEqual(list(trace.customdata[0][0]), ["Actual history 1", "H1", "Actual candidate 1", "C1"])

    def test_evidence_rejects_transposed_or_nonfinite_matrices(self):
        for values in ([[.1, .2]], [[.1, .2], [.3, math.inf]], [[1.1, .2], [.3, 0]]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                charts.evidence_heatmap(self.history, self.candidates, values)

    def test_network_keeps_negative_edges_distinct_and_filters_only_observed_edges(self):
        figure = charts.network_figure(self.history, self.candidates, self.values, min_abs=.3, max_edges=2)
        self.assertEqual(len(figure.data), 4)  # two measured edges + two node groups
        self.assertEqual([trace.customdata[0][2] for trace in figure.data[:2]], [.8, -.4])
        self.assertEqual([trace.line.dash for trace in figure.data[:2]], ["solid", "dot"])
        self.assertEqual([trace.marker.symbol for trace in figure.data[-2:]], ["diamond", "circle"])
        self.assertFalse(any(getattr(annotation, "showarrow", False) for annotation in figure.layout.annotations))
        self.assertIn("2/2 nonzero links", figure.layout.annotations[0].text)

    def test_network_and_heatmap_have_honest_empty_state(self):
        self.assertEqual(len(charts.evidence_heatmap([], [], []).data), 0)
        self.assertEqual(len(charts.network_figure([], [], []).data), 0)


class SeedMetricChartTests(unittest.TestCase):
    @staticmethod
    def rows():
        return [{"model": model, "seed": seed, "split": "test", "domain": "fixture-domain", "ndcg@10": value,
                 "items_sha256": "same items", "run_protocol": {"candidate_set": "full_catalog", "evaluation_subset": False}}
                for model, entries in (("qwen_frozen", [(42, .01), (43, .03), (44, .02)]),
                                       ("qwen_semantic", [(42, .02), (44, .01)]),
                                       ("qwen_preference", [(42, .04), (43, .02), (44, .03)]))
                for seed, value in entries]

    def test_seed_points_and_mean_are_actual_with_no_jitter_or_imputation(self):
        figure = charts.seed_metric_figure(self.rows(), "ndcg@10")
        self.assertEqual(len(figure.data), 4)
        self.assertEqual(list(figure.data[1].y), [.02, .01])
        self.assertEqual([row[0] for row in figure.data[1].customdata], [42, 44])
        self.assertEqual(len(set(figure.data[1].x)), 1)
        self.assertEqual(list(figure.data[-1].y), [.02, .015, .03])
        self.assertEqual([row[0] for row in figure.data[-1].customdata], [3, 2, 3])
        self.assertIsNone(figure.data[0].error_y.type)
        self.assertEqual(charts.seed_metric_figure(self.rows(), "ndcg@10").to_json(), figure.to_json())

    def test_paired_deltas_only_use_intersection_with_reference_seed(self):
        figure = charts.paired_delta_figure(self.rows(), "ndcg@10")
        self.assertEqual(list(figure.data[0].x), ["seed 42", "seed 44"])
        self.assertEqual(list(figure.data[0].y), [.01, -.01])
        self.assertEqual([row[0] for row in figure.data[0].customdata], [42, 44])
        self.assertAlmostEqual(figure.data[1].y[0], .03)
        self.assertAlmostEqual(figure.data[1].y[1], -.01)
        self.assertEqual(len(figure.layout.shapes), 1)
        self.assertEqual(figure.layout.shapes[0].y0, 0)

    def test_missing_reference_or_matching_seed_is_not_fabricated(self):
        rows = [row for row in self.rows() if row["model"] != "qwen_frozen"]
        self.assertEqual(len(charts.paired_delta_figure(rows, "ndcg@10").data), 0)
        rows = [self.rows()[0], {**self.rows()[3], "seed": 99}]
        self.assertEqual(len(charts.paired_delta_figure(rows, "ndcg@10").data), 0)

    def test_different_domains_protocols_or_duplicate_seeds_cannot_be_merged(self):
        changes = ({"domain": "different"}, {"items_sha256": "different"}, {"run_protocol": {"candidate_set": "sampled"}}, {"training_configuration": {"max_examples": 1}})
        for change in changes:
            rows = self.rows()
            rows[-1].update(change)
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "Cannot combine records"):
                charts.seed_metric_figure(rows, "ndcg@10")
        rows = self.rows() + [self.rows()[0]]
        with self.assertRaisesRegex(ValueError, "Duplicate seed"):
            charts.paired_delta_figure(rows, "ndcg@10")

    def test_non_test_subset_and_nonfinite_metrics_are_rejected(self):
        for change in ({"split": "validation"}, {"run_protocol": {"evaluation_subset": True}}, {"ndcg@10": math.nan}):
            rows = self.rows()
            rows[0].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                charts.seed_metric_figure(rows, "ndcg@10")


if __name__ == "__main__":
    unittest.main()
