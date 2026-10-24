"""Data protocol tests use explicit synthetic fixtures, never benchmark results."""
import gzip
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from shoplens.data import DataError, audit_images, download_file, iter_records, prepare, write_jsonl


class DataTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def test_loose_metadata_is_parsed_without_eval(self):
        path = self.root / "metadata.gz"
        with gzip.open(path, "wt") as stream:
            stream.write("{'asin':'A', 'description':['hello', 'world']}\n")
            stream.write('{"asin":"B"}\n')
        self.assertEqual([record["asin"] for record in iter_records(path)], ["A", "B"])
        with gzip.open(path, "wt") as stream:
            stream.write("__import__('os').system('touch forbidden')\n")
        with self.assertRaises(DataError):
            list(iter_records(path))

    def fixture(self):
        reviews = self.root / "reviews.jsonl"
        records = [{"reviewerID": f"u{number % 3}", "asin": f"i{number % 5}", "overall": 5,
                    "unixReviewTime": number, "reviewText": f"review {number}", "summary": ""}
                   for number in range(30)]
        write_jsonl(reviews, records)
        metadata = self.root / "meta.jsonl"
        # Reverse order checks that ASIN, rather than row position, joins data.
        write_jsonl(metadata, ({"asin": f"i{number}", "title": f"Product {number}",
                               "description": ["<b>plain</b>", "text &amp; more"],
                               "imUrl": f"https://example.invalid/{number}.jpg"}
                              for number in reversed(range(5))))
        return reviews, metadata

    def test_temporal_review_exclusion_and_id_join(self):
        reviews, metadata = self.fixture()
        processed = self.root / "processed"
        audit = prepare("baby", reviews, metadata, processed)
        items = list(iter_records(processed / "items.jsonl"))
        self.assertEqual([item["item_id"] for item in items], [f"i{i}" for i in range(5)])
        self.assertEqual(items[2]["title"], "Product 2")
        self.assertEqual(items[2]["text"], "Product 2 plain text & more")
        self.assertNotIn("review", items[2]["text"])
        cut = audit["cutoffs"]["train_end"]
        train_reviews = list(iter_records(processed / "reviews_train.jsonl"))
        self.assertTrue(all(record["timestamp"] <= cut for record in train_reviews))
        self.assertEqual(len(train_reviews), audit["counts"]["splits"]["train"])
        events = list(iter_records(processed / "events.jsonl"))
        self.assertTrue(all(event["timestamp"] > cut for event in events if event["split"] != "train"))
        self.assertEqual(set(audit["counts"]["splits"]), {"train", "validation", "test"})

    def test_subset_retains_all_selected_users_events(self):
        reviews, metadata = self.fixture()
        processed = self.root / "processed"
        audit = prepare("baby", reviews, metadata, processed, max_users=1)
        self.assertEqual(audit["scope"], "subset")
        self.assertEqual(audit["counts"]["users"], 1)
        self.assertEqual(audit["counts"]["events"], 10)

    def test_equal_timestamp_is_not_past_history(self):
        reviews, metadata = self.fixture()
        write_jsonl(reviews, ({"reviewerID": "u", "asin": f"i{i}", "overall": 5,
                              "unixReviewTime": 100, "reviewText": "same day"} for i in range(5)))
        audit = prepare("baby", reviews, metadata, self.root / "processed")
        self.assertEqual(audit["counts"]["positive_targets_with_prior_history"], {})

    def test_missing_metadata_is_explicit_or_filtered(self):
        reviews, metadata = self.fixture()
        write_jsonl(metadata, [{"asin": "i0", "title": "Only known product"}])
        audit = prepare("baby", reviews, metadata, self.root / "all")
        self.assertEqual(audit["counts"]["missing_metadata_before_filter"], 4)
        filtered = prepare("baby", reviews, metadata, self.root / "filtered", require_text=True, min_user_events=1)
        self.assertEqual(filtered["counts"]["items"], 1)
        self.assertEqual(filtered["counts"]["events"], 6)

    def test_image_audit_never_claims_unfetched_url_is_valid(self):
        processed = self.root / "processed"
        write_jsonl(processed / "items.jsonl", [
            {"item_id": "A", "image_url": "https://example.invalid/A.jpg", "image_path": ""},
            {"item_id": "B", "image_url": "", "image_path": ""},
        ])
        report = audit_images(processed, limit=None, download=False)
        self.assertEqual(report["counts"], {"not_downloaded": 1, "missing_url": 1})

    def test_verified_image_path_updates_catalog(self):
        processed = self.root / "processed"
        image = self.root / "A.jpg"
        image.write_bytes(b"explicit mocked image fixture")
        write_jsonl(processed / "items.jsonl", [{"item_id": "A", "image_url": "", "image_path": str(image)}])
        with patch("shoplens.data._validate_image"):
            report = audit_images(processed, limit=None)
        self.assertEqual(report["counts"], {"ok": 1})
        self.assertEqual(next(iter_records(processed / "items.jsonl"))["image_path"], str(image.resolve()))

    def response(self, payload, *, status=200, headers=None):
        result = io.BytesIO(payload)
        result.status = status
        result.headers = headers or {"Content-Length": str(len(payload))}
        return result

    def test_resume_only_appends_if_range_is_honored(self):
        destination = self.root / "data.bin"
        destination.with_name("data.bin.part").write_bytes(b"abc")
        response = self.response(b"def", status=206,
                                 headers={"Content-Length": "3", "Content-Range": "bytes 3-5/6"})
        with patch("shoplens.data.urllib.request.urlopen", return_value=response) as request:
            result = download_file("https://example.invalid/data", destination)
        self.assertEqual(destination.read_bytes(), b"abcdef")
        self.assertEqual(request.call_args.args[0].get_header("Range"), "bytes=3-")
        self.assertFalse(result["cached"])

    def test_range_ignored_replaces_partial(self):
        destination = self.root / "data.bin"
        destination.with_name("data.bin.part").write_bytes(b"old partial")
        with patch("shoplens.data.urllib.request.urlopen", return_value=self.response(b"complete")):
            download_file("https://example.invalid/data", destination)
        self.assertEqual(destination.read_bytes(), b"complete")

    def test_html_is_not_accepted_as_a_dataset(self):
        response = self.response(b"<html>access denied</html>", headers={"Content-Type": "text/html"})
        with patch("shoplens.data.urllib.request.urlopen", return_value=response):
            with self.assertRaisesRegex(DataError, "HTML"):
                download_file("https://example.invalid/data.gz", self.root / "data.gz", retries=1)
        self.assertFalse((self.root / "data.gz").exists())


if __name__ == "__main__":
    unittest.main()
