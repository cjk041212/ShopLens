"""Lightweight catalog/path and exact-row parity fixtures; no encoder loading."""

import importlib.util
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from shoplens import studio

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("shoplens_studio_benchmark_fixture", ROOT / "scripts/benchmark_studio.py")
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)
BASELINE = "eb53149346207b2500d8e160682e3894fb30125d"


class BenchmarkInputTests(unittest.TestCase):
    def test_catalog_images_follow_flat_or_nested_metadata_and_deduplicate_bytes(self):
        with tempfile.TemporaryDirectory() as work:
            root = Path(work)
            paths = ["data/images/first.jpg", "data/images/clothing/second.jpg", "data/images/clothing/duplicate.jpg"]
            for path, content in zip(paths, (b"DIAGNOSTIC ONE", b"DIAGNOSTIC TWO", b"DIAGNOSTIC ONE")):
                target = root / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
            items = [{"item_id": str(index), "image_path": path} for index, path in enumerate(paths)]
            images, evidence = benchmark.catalog_images(root, items, count=2)
            self.assertEqual(images, [b"DIAGNOSTIC ONE", b"DIAGNOSTIC TWO"])
            self.assertEqual([row["path"] for row in evidence], paths[:2])
            with self.assertRaises(FileNotFoundError):
                benchmark.catalog_images(root, items, count=3)

    def test_catalog_image_traversal_is_rejected(self):
        for path in ("../outside.jpg", "/absolute.jpg", "data/images/../../private", "other/first.jpg"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                benchmark.catalog_images("/private/tmp", [{"item_id": "fixture", "image_path": path}], 1)

    def test_timing_summary_keeps_samples_and_rejects_invalid_data(self):
        record = benchmark.stats([1., 2., 3.])
        self.assertEqual(record["n"], 3)
        self.assertEqual(record["p50_ms"], 2.)
        self.assertEqual(record["samples_ms"], [1., 2., 3.])
        for samples in ([], [float("nan")], [-1.]):
            with self.subTest(samples=samples), self.assertRaises(ValueError):
                benchmark.stats(samples)

    def test_baseline_requires_full_pinned_commit(self):
        for commit in ("main", "--format=anything", "ab12", "A" * 40):
            with self.subTest(commit=commit), self.assertRaises(ValueError):
                benchmark.load_baseline(commit)


class MetadataPoolParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.baseline, _ = benchmark.load_baseline(BASELINE)
        except (FileNotFoundError, subprocess.CalledProcessError) as error:
            raise unittest.SkipTest("Pinned public baseline Git object is unavailable offline") from error

    def setUp(self):
        rng = np.random.default_rng(42)
        self.items = [{"item_id": f"FIXTURE-{index}", "title": "Red leather bag" if index % 2 else "Blue cotton bag",
                       "text": "Durable travel bag", "metadata": {"price": None if index == 0 else 10 * index,
                                                                   "categories": [["Clothing", "Bags"]]}}
                      for index in range(12)]
        self.text = rng.standard_normal((12, 5)).astype(np.float32)
        self.images = rng.standard_normal((12, 3)).astype(np.float32)
        self.query = self.text[3].copy()

    def compare(self, **kwargs):
        options = {"query_vector": self.query, "candidate_pool_size": 4, "k": 8, **kwargs}
        before = self.baseline.rank_products(self.items, self.text, self.images, **options)
        after = studio.rank_products(self.items, self.text, self.images, **options)
        self.assertEqual(before, after)
        return after

    def test_complete_rows_match_public_baseline_for_three_workloads(self):
        for options in ({}, {"image_vector": self.images[2], "positive_ids": ["FIXTURE-1"],
                             "negative_ids": ["FIXTURE-5"], "personalization_scores": np.linspace(-1, 1, 12)},
                        {"avoid_terms": "leather; red", "max_price": 50.}):
            with self.subTest(options=tuple(options)):
                self.compare(**options)

    def test_cutoff_ties_and_category_filter_do_not_change_complete_rows(self):
        self.text[:] = 1.
        rows = self.compare(query_vector=np.ones(5), required_category="Clothing / Bags", avoid_terms="red", max_price=60.)
        self.assertTrue(rows)
        self.assertTrue(all(row["components"]["pool_count"] == 12 for row in rows))
        self.assertEqual(self.compare(required_category="Unavailable category"), [])

    def test_feedback_without_query_keeps_full_catalog_semantics(self):
        self.compare(query_vector=None, positive_ids=["FIXTURE-2"], hidden_ids=["FIXTURE-8"], avoid_terms="leather")


if __name__ == "__main__":
    unittest.main()
