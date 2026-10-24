"""Synthetic protocol fixtures; no model downloads or benchmark scores."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from shoplens.models import TemporalExamples, load_events, load_items
from shoplens.qwen_data import build_data, sample_negatives


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class QwenDataTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.processed = self.root / "data/processed/baby"
        self.items = [{"item_id": f"i{i}", "title": f"Product {i}", "text": f"Original description for product {i}"}
                      for i in range(9)]
        events = []
        for user in ("alice", "bob"):
            for item, timestamp, split, rating in (("i0", 1, "train", 5), ("i1", 2, "train", 5),
                ("i2", 2, "train", 5), ("i3", 3, "train", 5), ("i4", 4, "train", 5),
                ("i5", 5, "train", 3), ("i6", 11, "validation", 5), ("i7", 21, "test", 5)):
                events.append({"user_id": user, "item_id": item, "timestamp": timestamp,
                               "split": split, "rating": rating})
        self.events = events
        self.reviews = [{"user_id": row["user_id"], "item_id": row["item_id"],
                         "timestamp": row["timestamp"], "rating": row["rating"],
                         "text": f"{row['user_id']} exact review for {row['item_id']} at {row['timestamp']}"}
                        for row in events if row["split"] == "train"]
        write_jsonl(self.processed / "items.jsonl", self.items)
        write_jsonl(self.processed / "events.jsonl", events)
        write_jsonl(self.processed / "reviews_train.jsonl", self.reviews)
        (self.processed / "audit.json").write_text(json.dumps({"cutoffs": {"train_end": 10, "validation_end": 20}}))

    def test_case_protocol_matches_baseline_and_equal_time_is_not_history(self):
        data = build_data(self.root, "baby", 42)
        temporal = TemporalExamples(load_events(self.processed), [row["item_id"] for row in load_items(self.processed)])
        self.assertEqual(data.training_examples, temporal.examples("train", include_seen=False))
        self.assertEqual(data.validation_examples, temporal.examples("validation"))
        for case in data.adaptation_examples:
            if case["timestamp"] == 2:
                self.assertEqual(case["history"], [0])
            self.assertNotIn(case["target"], case["history"])
            self.assertTrue(case["review"].startswith(case["user_id"] + " exact review"))
            self.assertIn(f"for {data.item_ids[case['target']]} at {case['timestamp']}", case["review"])

    def test_same_seed_has_identical_case_order_hashes_and_negatives(self):
        first = build_data(self.root, "baby", 42, adaptation_limit=4)
        second = build_data(self.root, "baby", 42, adaptation_limit=4)
        other = build_data(self.root, "baby", 43, adaptation_limit=4)
        self.assertEqual(first.adaptation_examples, second.adaptation_examples)
        self.assertEqual(first.hashes, second.hashes)
        self.assertNotEqual(first.hashes["adaptation_cases_sha256"], other.hashes["adaptation_cases_sha256"])
        negative = sample_negatives(first.adaptation_examples, first.train_positive_sets, len(first.items), 101)
        np.testing.assert_array_equal(negative, sample_negatives(second.adaptation_examples, second.train_positive_sets,
                                                                len(second.items), 101))
        self.assertEqual(negative.dtype, np.int64)
        for case, item in zip(first.adaptation_examples, negative):
            self.assertNotIn(item, first.train_positive_sets[case["user_id"]])

    def test_review_must_match_user_item_and_timestamp(self):
        self.reviews[0]["user_id"] = "unknown-user"
        write_jsonl(self.processed / "reviews_train.jsonl", self.reviews)
        with self.assertRaisesRegex(ValueError, "matching.*user,item,timestamp"):
            build_data(self.root, "baby", 42)

    def test_future_review_is_rejected(self):
        self.reviews[0]["timestamp"] = 11
        write_jsonl(self.processed / "reviews_train.jsonl", self.reviews)
        with self.assertRaisesRegex(ValueError, "cutoff"):
            build_data(self.root, "baby", 42)

    def test_wrong_timestamp_within_training_period_is_rejected(self):
        self.reviews[0]["timestamp"] = 3
        write_jsonl(self.processed / "reviews_train.jsonl", self.reviews)
        with self.assertRaisesRegex(ValueError, "matching.*user,item,timestamp"):
            build_data(self.root, "baby", 42)

    def test_missing_review_skips_case_without_borrowing_another_users_text(self):
        retained = [review for review in self.reviews
                    if not (review["user_id"] == "alice" and review["item_id"] == "i3")]
        write_jsonl(self.processed / "reviews_train.jsonl", retained)
        data = build_data(self.root, "baby", 42)
        self.assertFalse(any(case["user_id"] == "alice" and case["target"] == 3
                             for case in data.adaptation_examples))
        self.assertTrue(any(case["user_id"] == "bob" and case["target"] == 3
                            for case in data.adaptation_examples))

    def test_review_rating_must_match_training_event(self):
        self.reviews[0]["rating"] = 4
        write_jsonl(self.processed / "reviews_train.jsonl", self.reviews)
        with self.assertRaisesRegex(ValueError, "rating does not match"):
            build_data(self.root, "baby", 42)

    def test_short_or_nonpositive_reviews_never_enter_adaptation(self):
        for review in self.reviews:
            if review["item_id"] == "i1":
                review["text"] = "too short"
        write_jsonl(self.processed / "reviews_train.jsonl", self.reviews)
        data = build_data(self.root, "baby", 42)
        self.assertTrue(data.adaptation_examples)
        self.assertTrue(all(case["target"] not in (1, 5) for case in data.adaptation_examples))
        self.assertEqual(data.sample_metadata["short_reviews"], 2)
        self.assertEqual(data.sample_metadata["nonpositive_reviews"], 2)

    def test_reviews_are_not_arbitrarily_chosen_by_asin(self):
        data = build_data(self.root, "baby", 42)
        alice = next(case for case in data.adaptation_examples if case["user_id"] == "alice" and case["target"] == 3)
        bob = next(case for case in data.adaptation_examples if case["user_id"] == "bob" and case["target"] == 3)
        self.assertNotEqual(alice["review"], bob["review"])
        self.assertNotEqual(alice["case_id"], bob["case_id"])

    def test_conflicting_duplicate_reviews_fail(self):
        duplicate = {**self.reviews[1], "text": "A different review for exactly the same event"}
        write_jsonl(self.processed / "reviews_train.jsonl", self.reviews + [duplicate])
        with self.assertRaisesRegex(ValueError, "Conflicting reviews"):
            build_data(self.root, "baby", 42)

    def test_validation_test_likes_are_not_in_train_negative_exclusions(self):
        data = build_data(self.root, "baby", 42)
        self.assertEqual(data.train_positive_sets["alice"], {0, 1, 2, 3, 4})
        negatives = sample_negatives(data.adaptation_examples * 30, data.train_positive_sets, 9, 7)
        self.assertTrue(set(negatives).issubset({5, 6, 7, 8}))
        self.assertIn(6, negatives)
        self.assertIn(7, negatives)

    def test_dense_user_without_negative_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "No train-unobserved negative"):
            sample_negatives([{"user_id": "all", "target": 0}], {"all": {0, 1}}, 2, 42)

    def test_max_history_and_input_files_are_preserved(self):
        files = list(self.processed.glob("*.json*"))
        original = {path: path.read_bytes() for path in files}
        data = build_data(self.root, "baby", 42, max_history=2)
        self.assertTrue(all(len(case["history"]) <= 2 for case in data.adaptation_examples))
        self.assertEqual(data.hashes["items_sha256"].__len__(), 64)
        self.assertEqual(original, {path: path.read_bytes() for path in files})

    def test_mislabelled_prepared_split_is_rejected(self):
        self.events[0]["split"] = "validation"
        write_jsonl(self.processed / "events.jsonl", self.events)
        with self.assertRaises(ValueError):
            build_data(self.root, "baby", 42)


if __name__ == "__main__":
    unittest.main()
