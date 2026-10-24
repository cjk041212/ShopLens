"""Deterministic tiny fixtures; no models, network, training or formal evaluation."""

from __future__ import annotations

import copy
import csv
import importlib.util
import io
import json
from pathlib import Path
import unittest

import numpy as np

MODULE_PATH = Path(__file__).resolve().parents[1] / "src/shoplens/studio.py"
spec = importlib.util.spec_from_file_location("shoplens_studio_fixture", MODULE_PATH)
studio = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(studio)


class BriefRuleTests(unittest.TestCase):
    def test_budget_and_avoid_are_removed_from_positive_query(self):
        result = studio.parse_brief("Lightweight hiking backpack under $100; avoid leather and bright red.")
        self.assertEqual(result["query"], "Lightweight hiking backpack")
        self.assertEqual(result["avoid"], "leather and bright red")
        self.assertEqual(result["max_price"], 100.)
        self.assertIn("Hiking", result["tags"])
        self.assertEqual(result["parser"], "transparent_rules")
        self.assertIn("review and edit", result["notes"][0])

    def test_english_negation_markers_are_transparent_and_case_insensitive(self):
        for prefix in ("avoid", "no", "not", "don't like", "do not like", "without", "I don’t like"):
            with self.subTest(prefix=prefix):
                result = studio.parse_brief(f"Yoga mat; {prefix} neon; budget: $50")
                self.assertEqual(result["avoid"], "neon")
                self.assertEqual(result["query"], "Yoga mat")
                self.assertEqual(result["max_price"], 50.)

    def test_negation_stops_at_explicit_positive_clause(self):
        result = studio.parse_brief("I don't like neon, prefer blue shoes below 80 dollars")
        self.assertEqual(result["avoid"], "neon")
        self.assertEqual(result["query"], "prefer blue shoes")
        self.assertEqual(result["max_price"], 80.)

    def test_multiple_avoid_clauses_and_budgets_keep_strictest_editable_ceiling(self):
        result = studio.parse_brief("Travel bag; no leather and no red; under $120; $90 budget")
        self.assertEqual(result["avoid"], "leather; red")
        self.assertEqual(result["max_price"], 90.)
        self.assertTrue(any("lowest ceiling" in note for note in result["notes"]))

    def test_no_slip_and_not_only_are_not_misparsed_as_negative_preferences(self):
        result = studio.parse_brief("No-slip yoga mat, not only for exercise")
        self.assertEqual(result["avoid"], "")
        self.assertEqual(result["query"], "No-slip yoga mat, not only for exercise")
        self.assertIsNone(result["max_price"])

    def test_empty_or_dollar_only_briefs_are_explicit_not_llm_output(self):
        empty = studio.parse_brief("")
        self.assertEqual((empty["query"], empty["avoid"], empty["max_price"]), ("", "", None))
        self.assertEqual(studio.parse_brief("up to $1,000")['max_price'], 1000.)
        amount = studio.parse_brief("$75 backpack")
        self.assertEqual(amount["query"], "backpack")
        self.assertEqual(amount["max_price"], 75.)
        self.assertTrue(any("without a budget phrase" in note for note in amount["notes"]))


class StudioRankingTests(unittest.TestCase):
    def setUp(self):
        self.items = [
            {"item_id": "D", "title": "Blue trail bag", "description": "Small blue fabric bag for hiking.", "metadata": {"brand": "RealBrand", "price": 50., "categories": [["Sports", "Bags"]]}},
            {"item_id": "B", "title": "Red leather bag", "description": "Red leather exterior with a fabric lining.", "metadata": {"price": 35.}},
            {"item_id": "A", "title": "Green cotton bag", "description": "Cotton bag with reinforced handles.", "metadata": {"price": "$60.00"}},
            {"item_id": "C", "title": "Canvas carrier", "description": "Lightweight canvas carrier.", "metadata": {}},
        ]
        # Different dimensions stand in for independent real Qwen1024 / CLIP512 spaces.
        self.text = np.array([[1., 0., 0.], [.8, .6, 0.], [0., 1., 0.], [-1., 0., 0.]], dtype=np.float32)
        self.image = np.array([[0., 1.], [1., 0.], [.6, .8], [-1., 0.]], dtype=np.float32)
        self.query = np.array([1., 0., 0.], dtype=np.float32)
        self.image_query = np.array([1., 0.], dtype=np.float32)

    def rank(self, **kwargs):
        options = dict(query_vector=self.query, k=10)
        options.update(kwargs)
        return studio.rank_products(self.items, self.text, self.image, **options)

    @staticmethod
    def by_id(rows):
        return {row["item_id"]: row for row in rows}

    def test_independent_spaces_normalize_and_missing_image_has_full_text_weight(self):
        first = self.rank(text_weight=.1)
        second = self.rank(text_weight=.9)
        self.assertEqual([(row["item_id"], row["score"]) for row in first],
                         [(row["item_id"], row["score"]) for row in second])
        scaled = studio.rank_products(self.items, self.text * np.array([2., 3., 4., 5.])[:, None],
                                      self.image * 9, query_vector=self.query * 7, k=10)
        self.assertEqual([row["item_id"] for row in first], [row["item_id"] for row in scaled])
        np.testing.assert_allclose([row["score"] for row in first], [row["score"] for row in scaled])
        self.assertAlmostEqual(first[0]["components"]["query_text_cosine"], 1.)
        self.assertIsNone(first[0]["components"]["query_image_cosine"])

    def test_query_modalities_fuse_percentiles_instead_of_raw_cosines(self):
        rows = self.rank(image_vector=self.image_query, text_weight=.65)
        for row in rows:
            parts = row["components"]
            self.assertAlmostEqual(row["score"], .65 * parts["query_text_percentile"] + .35 * parts["query_image_percentile"])
        image_only = self.rank(query_vector=None, image_vector=self.image_query, text_weight=1.)
        self.assertEqual(image_only[0]["item_id"], "B")
        self.assertEqual(image_only[0]["score"], 1.)

    def test_ties_use_average_percentile_and_stable_item_id_order(self):
        tied = np.ones_like(self.text)
        rows = studio.rank_products(self.items, tied, None, query_vector=np.ones(3), k=10)
        self.assertEqual([row["item_id"] for row in rows], ["A", "B", "C", "D"])
        self.assertEqual([row["score"] for row in rows], [.5] * 4)
        np.testing.assert_allclose(studio._percentiles([0., 0., 2., 4.]), [1/6, 1/6, 2/3, 1.])

    def test_hidden_positive_and_negative_are_all_hard_exclusions(self):
        rows = self.rank(positive_ids=["D"], negative_ids=["B"], hidden_ids=["A"])
        self.assertEqual([row["item_id"] for row in rows], ["C"])
        self.assertIn("negative_penalty", rows[0]["components"])
        with self.assertRaises(ValueError):
            self.rank(positive_ids=["D"], negative_ids=["D"])

    def test_hiding_or_budget_filtering_does_not_rescale_existing_scores(self):
        base = self.by_id(self.rank())
        hidden = self.by_id(self.rank(hidden_ids=["D"]))
        budget = self.by_id(self.rank(max_price=50))
        for item_id, row in hidden.items():
            self.assertEqual(row["score"], base[item_id]["score"])
        self.assertEqual(set(budget), {"B", "D"})
        for item_id, row in budget.items():
            self.assertEqual(row["score"], base[item_id]["score"])

    def test_budget_uses_real_metadata_and_excludes_unknown_or_range_prices(self):
        self.items[0]["price"] = 1.  # Nested prepared metadata remains authoritative.
        rows = self.rank(max_price=35)
        self.assertEqual([row["item_id"] for row in rows], ["B"])
        self.items[3]["metadata"]["price"] = "$10 - $20"
        self.assertNotIn("C", self.by_id(self.rank(max_price=100)))
        self.assertEqual(self.by_id(self.rank())["A"]["components"]["historical_price"], 60.)
        self.assertTrue(any("not a live quote" in fact for fact in rows[0]["evidence"]))

    def test_positive_or_personalized_fallback_is_allowed_but_no_signal_is_empty(self):
        self.assertEqual(self.rank(query_vector=None), [])
        self.assertEqual(self.rank(query_vector=None, negative_ids=["D"]), [])
        positive = self.rank(query_vector=None, positive_ids=["D"])
        self.assertEqual(len(positive), 3)
        self.assertTrue(all(row["components"]["feedback_score"] is not None for row in positive))
        personal = self.rank(query_vector=None, personalization_scores={"D": 0., "B": 3., "A": 2., "C": 1.})
        self.assertEqual(personal[0]["item_id"], "B")
        self.assertEqual(personal[0]["score"], 1.)

    def test_query_feedback_blend_uses_requested_weight_and_separate_penalties(self):
        rows = self.rank(personalization_scores=[0., 1., 2., 3.], feedback_weight=.3,
                         negative_ids=["B"], avoid_vector=self.query, avoid_terms=["leather"])
        for row in rows:
            part = row["components"]
            expected_base = .7 * part["query_score"] + .3 * part["feedback_score"]
            self.assertAlmostEqual(part["base_score"], expected_base)
            expected = expected_base - part["negative_penalty"] - part["avoid_semantic_penalty"] - part["avoid_terms_penalty"]
            self.assertAlmostEqual(row["score"], expected)
        self.assertNotIn("B", self.by_id(rows))

    def test_avoid_hits_are_soft_literal_word_matches_with_original_evidence(self):
        rows = self.by_id(self.rank(avoid_terms="LEATHER; cotton"))
        self.assertIn("B", rows)  # Soft avoidance is not a fabricated hard filter.
        self.assertEqual(rows["B"]["components"]["avoid_term_hits"], ["leather"])
        self.assertEqual(rows["B"]["components"]["avoid_terms_penalty"], .2)
        self.assertTrue(any("Red leather exterior" in fact or "Red leather bag" in fact for fact in rows["B"]["evidence"]))
        self.items[1]["title"] = "Leatherette bag"
        self.items[1]["description"] = "Synthetic leatherette exterior."
        rows = self.by_id(self.rank(avoid_terms=["leather"]))
        self.assertEqual(rows["B"]["components"]["avoid_term_hits"], [])

    def test_explicit_source_category_matches_any_complete_nested_path_ignoring_case(self):
        self.items[1]["metadata"]["categories"] = [["Bags"], ["Sports", "Bags"]]
        rows = self.rank(required_category="sports / bags")
        self.assertEqual({row["item_id"] for row in rows}, {"D", "B"})
        for row in rows:
            self.assertEqual(row["components"]["required_category"], "sports / bags")
            self.assertIn("Sports / Bags", row["components"]["source_category_paths"])
            self.assertIn("Source category: Sports / Bags", row["evidence"])
        self.assertEqual([row["item_id"] for row in self.rank(required_category="Bags")], ["B"])
        self.assertEqual(self.rank(required_category="Sports"), [])

    def test_category_missing_malformed_or_unknown_path_does_not_match(self):
        self.items[1]["metadata"]["categories"] = ["Sports", "Bags"]  # Not a nested original path.
        self.items[2]["metadata"]["categories"] = [["Sports", None, "Bags"]]
        rows = self.rank(required_category="Sports / Bags")
        self.assertEqual([row["item_id"] for row in rows], ["D"])
        self.assertEqual(self.rank(required_category="Unknown / Missing"), [])
        self.assertEqual(self.rank(required_category=""), [])
        with self.assertRaises(ValueError):
            self.rank(required_category=["Sports", "Bags"])

    def test_category_filter_does_not_rescale_scores_or_mutate_original_metadata(self):
        original = copy.deepcopy(self.items)
        baseline = self.by_id(self.rank())
        filtered = self.rank(required_category="Sports / Bags")
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["score"], baseline["D"]["score"])
        self.assertEqual(filtered[0]["components"]["query_text_percentile"], baseline["D"]["components"]["query_text_percentile"])
        self.assertEqual(self.items, original)

    def test_output_is_finite_json_serializable_and_preserves_catalog(self):
        original = copy.deepcopy(self.items)
        rows = self.rank(image_vector=self.image_query, positive_ids=["A"], negative_ids=["B"], avoid_terms=["blue"])
        encoded = json.dumps(rows, allow_nan=False)
        self.assertTrue(encoded)
        self.assertEqual(self.items, original)
        self.assertEqual([row["rank"] for row in rows], list(range(1, len(rows) + 1)))
        self.assertTrue(all(row["score_kind"] == "studio_rank_percentile_fusion" for row in rows))

    def test_invalid_shapes_nonfinite_unknown_ids_and_weights_reject(self):
        for arguments in (dict(query_vector=[0, 0, 0]), dict(query_vector=[1, 0]), dict(query_vector=[np.nan, 0, 0]),
                          dict(text_weight=-.1), dict(feedback_weight=np.nan), dict(max_price=-1),
                          dict(hidden_ids=["missing"]), dict(k=-1), dict(k=1.5),
                          dict(personalization_scores={"D": 1.}), dict(personalization_scores=[0, 1, np.inf, 2])):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                self.rank(**arguments)
        with self.assertRaises(ValueError):
            studio.rank_products(self.items, self.text[:2], self.image, query_vector=self.query)
        with self.assertRaises(ValueError):
            studio.rank_products(self.items, self.text, None, image_vector=self.image_query)

    def test_zero_catalog_vectors_never_produce_nan_and_zero_k_is_empty(self):
        rows = studio.rank_products(self.items, np.zeros_like(self.text), None, query_vector=self.query, k=10)
        self.assertEqual([row["score"] for row in rows], [.5] * 4)
        self.assertEqual(self.rank(k=0), [])
        self.assertEqual(studio.rank_products([], np.empty((0, 3)), None), [])


class RetrievalPoolTests(unittest.TestCase):
    def setUp(self):
        count = 301
        cosine = np.linspace(1., -.2, count, dtype=np.float32)
        self.text = np.column_stack((cosine, np.sqrt(1 - cosine ** 2)))
        self.image = np.tile([0., 1.], (count, 1)).astype(np.float32)
        self.image[200:] = [1., 0.]
        self.items = [{"item_id": f"FIXTURE-{i:03d}", "title": "Pink fixture" if i < 200 else "Blue fixture",
                       "metadata": {"price": 30. if i < 200 else 1.}}
                      for i in range(count)]

    def rank(self, **kwargs):
        options = dict(query_vector=[1., 0.], image_vector=[1., 0.], text_weight=.01,
                       avoid_vector=[1., 0.], avoid_terms="pink", negative_ids=["FIXTURE-000"],
                       max_price=20, k=12)
        options.update(kwargs)
        return studio.rank_products(self.items, self.text, self.image, **options)

    def test_budget_and_strong_reranking_cannot_admit_products_outside_text_query_pool(self):
        self.assertEqual(self.rank(), [])
        diagnostic = self.rank(candidate_pool_size=None)
        self.assertEqual(len(diagnostic), 12)
        self.assertTrue(all(int(row["item_id"][-3:]) >= 200 for row in diagnostic))
        self.items[199]["metadata"]["price"] = 10.
        limited = self.rank()
        self.assertEqual([row["item_id"] for row in limited], ["FIXTURE-199"])
        self.assertEqual(limited[0]["components"]["pool_count"], 200)
        self.assertEqual(limited[0]["components"]["pool_channel"], "text_query")
        self.assertTrue(any("not a relevance guarantee" in reason for reason in limited[0]["reasons"]))

    def test_category_filter_stays_inside_fixed_query_pool(self):
        for item in self.items[200:]:
            item["metadata"]["categories"] = [["Fixture", "Explicit category"]]
        self.assertEqual(self.rank(required_category="Fixture / Explicit category"), [])
        self.items[199]["metadata"].update(price=10., categories=[["Fixture", "Explicit category"]])
        rows = self.rank(required_category="Fixture / Explicit category")
        self.assertEqual([row["item_id"] for row in rows], ["FIXTURE-199"])
        self.assertEqual(rows[0]["components"]["pool_count"], 200)

    def test_cutoff_ties_are_all_retained_and_order_is_stable_by_id(self):
        # 199 strictly closer products, three equal products spanning the cutoff.
        self.text[199:202] = [0., 1.]
        self.text[202:] = [-1., 0.]
        rows = self.rank(image_vector=None, avoid_vector=None, avoid_terms=(), negative_ids=(),
                         max_price=None, text_weight=1., k=301)
        self.assertEqual(len(rows), 202)
        self.assertTrue(all(row["components"]["pool_count"] == 202 for row in rows))
        tied_ids = [row["item_id"] for row in rows if row["components"]["query_text_cosine"] == 0.]
        self.assertEqual(tied_ids, ["FIXTURE-199", "FIXTURE-200", "FIXTURE-201"])

    def test_pool_gating_does_not_rescale_scores_or_change_with_filters(self):
        base_options = dict(image_vector=None, avoid_vector=None, avoid_terms=(), negative_ids=(),
                            max_price=None, text_weight=1., k=301)
        gated = self.rank(**base_options)
        diagnostic = {row["item_id"]: row for row in self.rank(**base_options, candidate_pool_size=None)}
        self.assertEqual(len(gated), 200)
        for row in gated:
            self.assertEqual(row["score"], diagnostic[row["item_id"]]["score"])
            self.assertEqual(row["components"]["query_text_percentile"], diagnostic[row["item_id"]]["components"]["query_text_percentile"])
        filtered_options = {**base_options, "hidden_ids": ["FIXTURE-000"], "max_price": 30}
        filtered = self.rank(**filtered_options)
        self.assertEqual(len(filtered), 199)
        self.assertTrue(all(row["components"]["pool_count"] == 200 for row in filtered))
        self.assertNotIn("FIXTURE-200", {row["item_id"] for row in filtered})

    def test_image_only_gate_and_feedback_only_full_catalog(self):
        self.image = self.text[::-1].copy()
        image_only = self.rank(query_vector=None, avoid_vector=None, avoid_terms=(), negative_ids=(),
                               max_price=None, k=301)
        self.assertEqual(len(image_only), 200)
        self.assertTrue(all(int(row["item_id"][-3:]) >= 101 for row in image_only))
        self.assertEqual(image_only[0]["components"]["pool_channel"], "image_query")
        feedback = self.rank(query_vector=None, image_vector=None, avoid_vector=None, avoid_terms=(),
                             negative_ids=(), positive_ids=["FIXTURE-000"], max_price=None, k=301)
        self.assertEqual(len(feedback), 300)
        self.assertTrue(any(int(row["item_id"][-3:]) >= 200 for row in feedback))
        self.assertEqual(feedback[0]["components"]["pool_channel"], "feedback_only")

    def test_invalid_pool_sizes_reject(self):
        for size in (0, -1, True, 1.5):
            with self.subTest(size=size), self.assertRaises(ValueError):
                self.rank(candidate_pool_size=size)


class ShortlistExportTests(unittest.TestCase):
    def setUp(self):
        self.items = [{"item_id": "B", "title": "Original, title", "description": "Original\nproduct text", "metadata": {"brand": "RealBrand", "price": 12., "categories": [["Sports"]]}, "image_url": "https://example.invalid/original.jpg"},
                      {"item_id": "A", "title": "Second", "text": "Full original text", "metadata": {}}]
        self.source = {"dataset": "Synthetic fixture only", "metadata": "prepared items.jsonl"}

    def test_json_keeps_original_metadata_sources_and_selected_order_without_mutation(self):
        original = copy.deepcopy(self.items)
        brief = {"query": "fixture only", "avoid": "", "max_price": 20}
        ranked = [{"item_id": "B", "rank": 1, "score": .5, "components": {"query_score": .5}, "evidence": ["Original fact"]}]
        result = studio.export_shortlist(self.items, ["A", "B", "A"], brief, recommendations=ranked, source=self.source)
        self.assertEqual(result["selected_item_ids"], ["A", "B"])
        self.assertEqual(result["items"][1]["metadata"], self.items[0]["metadata"])
        self.assertEqual(result["items"][1]["original_text"], "Original\nproduct text")
        self.assertEqual(result["items"][1]["studio_recommendation"], ranked[0])
        decoded = json.loads(studio.shortlist_json(self.items, ["B"], brief, recommendations=ranked, source=self.source))
        self.assertEqual(decoded["source"], self.source)
        self.assertEqual(decoded["items"][0]["historical_price"], 12.)
        result["items"][1]["metadata"]["price"] = 999
        self.assertEqual(self.items, original)

    def test_csv_formula_prefixes_are_escaped_only_in_the_export_copy(self):
        for prefix in ("=", "+", "-", "@", " \t="):
            with self.subTest(prefix=prefix):
                self.items[0]["title"] = prefix + "FORMULA()"
                self.items[0]["description"] = prefix + "Original text"
                original = copy.deepcopy(self.items)
                rows = list(csv.DictReader(io.StringIO(studio.shortlist_csv(self.items, ["B"], "brief"))))
                self.assertEqual(rows[0]["title"], "'" + original[0]["title"])
                self.assertEqual(rows[0]["original_text"], "'" + original[0]["description"])
                decoded = json.loads(studio.shortlist_json(self.items, ["B"], "brief"))
                self.assertEqual(decoded["items"][0]["title"], original[0]["title"])
                self.assertEqual(self.items, original)

    def test_csv_preserves_quoted_text_and_explicit_unknown_price(self):
        text = studio.shortlist_csv(self.items, ["B", "A"], "Original brief", source=self.source)
        rows = list(csv.DictReader(io.StringIO(text)))
        self.assertEqual(rows[0]["title"], "Original, title")
        self.assertEqual(rows[0]["original_text"], "Original\nproduct text")
        self.assertEqual(json.loads(rows[0]["source"]), self.source)
        self.assertEqual(rows[1]["historical_price"], "")
        self.assertIn("Historical metadata price", rows[1]["price_note"])
        with self.assertRaises(ValueError):
            studio.export_shortlist(self.items, ["missing"], "brief")


if __name__ == "__main__":
    unittest.main()
