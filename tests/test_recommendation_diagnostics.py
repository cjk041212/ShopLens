"""Synthetic protocol fixtures; no access to original events or model files."""

import copy
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from shoplens.models import TemporalExamples, training_counts
from shoplens.recommendation_diagnostics import (
    BUCKETS, DOMAINS, HISTORY_CAPS, MODELS, evaluate_checkpoint, exposure_metrics,
    load_recommendation_diagnostics, paired_user_delta, popularity_bucket,
    select_user_cohort, validate_diagnostics,
)


class DiagnosticScorer:
    max_history = 20
    item_ids = [f"FIXTURE-{index:02d}" for index in range(25)]
    items = [{"item_id": value, "title": "Synthetic protocol product"} for value in item_ids]

    def score_histories(self, histories):
        result = []
        for history in histories:
            scores = -np.arange(25, dtype=float)
            scores[[24, 0, 1]] = [100, 90, 80]
            scores[2], scores[3] = (50, 40) if len(history) >= 2 else (40, 50)
            result.append(scores)
        return np.array(result)


class RecommendationDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.cases = [
            {"user_id": "PRIVATE-FIXTURE-A", "target": 2, "history": [1, 0], "seen": {0, 1, 24}},
            {"user_id": "PRIVATE-FIXTURE-A", "target": 3, "history": [2, 1, 0], "seen": {0, 1, 2, 24}},
            {"user_id": "PRIVATE-FIXTURE-B", "target": 2, "history": [1, 0], "seen": {0, 1, 24}},
        ]
        self.counts = np.zeros(25, dtype=np.int64)
        self.counts[1:4] = [1, 6, 21]

    def evaluate(self):
        return evaluate_checkpoint(DiagnosticScorer(), self.cases, self.counts, batch_size=2, bootstrap_repeats=20)

    def test_history_caps_keep_full_seen_exclusions_and_user_macro_average(self):
        original = copy.deepcopy(self.cases)
        curves, cohorts, exposure = self.evaluate()
        discounted = 1 / math.log2(3)
        expected = (((discounted + 1) / 2) + discounted) / 2
        self.assertAlmostEqual(curves[0]["ndcg@10"], expected)
        self.assertEqual(curves[-1]["ndcg@10"], 1.)
        self.assertEqual([row["history_cap"] for row in curves], list(HISTORY_CAPS))
        self.assertTrue(all(row["n_users"] == 2 and row["n_targets"] == 3 for row in curves))
        self.assertEqual(curves[0]["changed_history_targets"], 3)
        self.assertEqual(curves[-1]["changed_history_targets"], 0)
        self.assertAlmostEqual(curves[0]["paired_ndcg_delta_vs_cap20"]["mean"], expected - 1)
        self.assertEqual(self.cases, original)
        self.assertEqual(exposure["recommendation_slots"], 30)
        self.assertNotIn("FIXTURE-24", {row["item_id"] for row in exposure["most_exposed_products"]})
        self.assertEqual(cohorts[0]["n_targets"], 0)
        self.assertIsNone(cohorts[0]["ndcg@10"])
        self.assertEqual(cohorts[2]["n_targets"], 2)

    def test_cohort_samples_users_and_retains_all_of_their_targets(self):
        selected, eligible = select_user_cohort(self.cases, 1, seed=42)
        self.assertEqual(eligible, 2)
        selected_users = {row["user_id"] for row in selected}
        self.assertEqual(len(selected_users), 1)
        self.assertEqual(selected, [row for row in self.cases if row["user_id"] in selected_users])
        reversed_selected, _ = select_user_cohort(list(reversed(self.cases)), 1, seed=42)
        self.assertEqual(selected_users, {row["user_id"] for row in reversed_selected})
        self.assertEqual(select_user_cohort(self.cases, 100)[0], self.cases)

    def test_temporal_cases_do_not_use_equal_time_or_future_feedback(self):
        events = [
            {"user_id": "u", "item_id": "a", "timestamp": 1, "rating": 5, "split": "train"},
            {"user_id": "u", "item_id": "b", "timestamp": 2, "rating": 5, "split": "test"},
            {"user_id": "u", "item_id": "c", "timestamp": 2, "rating": 5, "split": "test"},
            {"user_id": "u", "item_id": "d", "timestamp": 3, "rating": 5, "split": "test"},
        ]
        cases = TemporalExamples(events, ["a", "b", "c", "d"], 20).examples("test")
        self.assertEqual(cases[0]["seen"], {0})
        self.assertEqual(cases[1]["seen"], {0})
        self.assertEqual(cases[2]["seen"], {0, 1, 2})

    def test_popularity_counts_only_distinct_positive_training_users(self):
        events = [{"user_id": "u", "item_id": "a", "split": split, "rating": rating}
                  for split, rating in (("train", 5), ("train", 5), ("validation", 5), ("test", 5))]
        events.extend([{"user_id": "other", "item_id": "a", "split": "train", "rating": 2},
                       {"user_id": "other", "item_id": "b", "split": "train", "rating": 4}])
        np.testing.assert_array_equal(training_counts(events, ["a", "b"]), [1, 1])
        self.assertEqual([popularity_bucket(value) for value in (0, 1, 5, 6, 20, 21)], ["0", "1-5", "1-5", "6-20", "6-20", "21+"])
        with self.assertRaises(ValueError):
            popularity_bucket(-1)

    def test_exposure_includes_zero_exposure_catalog_items(self):
        items = [{"item_id": str(index)} for index in range(4)]
        concentrated = exposure_metrics([10, 0, 0, 0], [0, 1, 6, 21], items)
        self.assertEqual(concentrated["coverage@10"], .25)
        self.assertAlmostEqual(concentrated["gini"], .75)
        self.assertEqual(concentrated["hhi"], 1.)
        uniform = exposure_metrics([1, 1, 1, 1], [0, 1, 6, 21], items)
        self.assertAlmostEqual(uniform["gini"], 0.)
        self.assertEqual(uniform["hhi"], .25)
        self.assertEqual(sum(row["exposure_share"] for row in uniform["popularity_exposure"]), 1.)

    def test_paired_bootstrap_requires_matching_users_and_is_reproducible(self):
        interval = paired_user_delta({"a": .5, "b": .75}, {"a": .25, "b": .5}, repeats=30)
        self.assertEqual(interval["mean"], .25)
        self.assertEqual(interval["ci95_low"], .25)
        self.assertEqual(interval["ci95_high"], .25)
        with self.assertRaises(ValueError):
            paired_user_delta({"a": 1.}, {"b": 1.})

    def test_invalid_history_and_nonfinite_score_are_rejected(self):
        self.cases[0]["seen"].add(2)
        with self.assertRaisesRegex(ValueError, "unseen-target"):
            self.evaluate()
        self.cases[0]["seen"].remove(2)
        class NonfiniteScorer(DiagnosticScorer):
            def score_histories(self, histories):
                return np.full((len(histories), 25), np.nan)
        with self.assertRaisesRegex(ValueError, "invalid full-catalog"):
            evaluate_checkpoint(NonfiniteScorer(), self.cases, self.counts, bootstrap_repeats=1)

    def snapshot(self):
        curves, cohorts, exposure = self.evaluate()
        result = {"schema_version": 1, "complete": True, "kind": "held_out_recommendation_diagnostics",
                  "protocol": {"checkpoint_seed": 42, "models": list(MODELS), "domains": list(DOMAINS), "history_caps": list(HISTORY_CAPS)},
                  "domains": [{"domain": domain, "catalog_items": 25, "selected_users": 2, "selected_targets": 3} for domain in DOMAINS],
                  "history_curves": [], "popularity_cohorts": [], "exposure": []}
        for domain in DOMAINS:
            for model in MODELS:
                identity = {"domain": domain, "model": model, "seed": 42}
                result["history_curves"].extend({**identity, **row} for row in curves)
                result["popularity_cohorts"].extend({**identity, **row} for row in cohorts)
                result["exposure"].append({**identity, **exposure})
        return result

    def test_aggregate_output_contains_no_private_cases_and_loader_rejects_partial_data(self):
        result = self.snapshot()
        encoded = json.dumps(result)
        self.assertNotIn("PRIVATE-FIXTURE", encoded)
        self.assertNotIn('"user_id"', encoded)
        with tempfile.TemporaryDirectory() as work:
            path = Path(work) / "diagnostics.json"
            path.write_text(encoded)
            self.assertEqual(load_recommendation_diagnostics(path), result)
        result["history_curves"].pop()
        with self.assertRaisesRegex(ValueError, "Incomplete diagnostic rows"):
            validate_diagnostics(result)

    def test_loader_rejects_duplicate_identities_private_fields_and_nan(self):
        for mutation in (lambda value: value["history_curves"].__setitem__(0, value["history_curves"][1]),
                         lambda value: value.update(user_id="PRIVATE"),
                         lambda value: value["exposure"][0].update(gini=float("nan"))):
            result = self.snapshot()
            mutation(result)
            with self.assertRaises(ValueError):
                validate_diagnostics(result)

    def test_loader_rejects_changed_cohort_or_impossible_coverage(self):
        for mutation in (lambda value: value["history_curves"][0].update(n_targets=100),
                         lambda value: value["popularity_cohorts"][0].update(n_targets=1, n_users=1, **{key: 0. for key in ("recall@10", "ndcg@10", "recall@20", "ndcg@20")}),
                         lambda value: value["exposure"][0].update(**{"coverage@10": 1.5})):
            result = self.snapshot()
            mutation(result)
            with self.assertRaises(ValueError):
                validate_diagnostics(result)


if __name__ == "__main__":
    unittest.main()
