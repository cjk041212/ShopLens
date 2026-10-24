"""Protocol/ranking invariants for the bounded, measured FashionIQ experiment."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from shoplens.fashion_retrieval import (METHODS, _composition_assets, build_fusion, compose_fashion_query,
    enforce_image_hash_disjoint, fusion_inputs, image_ids, load_fashion_results, normalized, rank_targets,
    retrieval_metrics, select_pairs, stable_order)

SOURCE_ROOT = Path(__file__).resolve().parents[1]


class FashionRetrievalTests(unittest.TestCase):
    def test_reference_is_excluded_before_target_ranking(self):
        # The reference would rank first without exclusion; the target must be1.
        ranks, rankings = rank_targets([[1., 0.]], [[1., 0.], [.9, .1], [0., 1.]],
                                      ["reference", "target", "other"],
                                      [{"reference": "reference", "target": "target"}])
        self.assertEqual(ranks, [1])
        self.assertEqual(rankings, [["target", "other"]])

    def test_target_cannot_be_missing_or_reference(self):
        for target in ("missing", "ref"):
            with self.subTest(target=target), self.assertRaises(ValueError):
                rank_targets([[1., 0.]], [[1., 0.], [0., 1.]], ["ref", "other"],
                             [{"reference": "ref", "target": target}])

    def test_tie_break_does_not_depend_on_candidate_order(self):
        vectors = [[1., 0.], [1., 0.], [0., 1.]]
        rows = [{"reference": "absent", "target": "a"}]
        one = rank_targets([[1., 0.]], vectors, ["b", "a", "c"], rows)
        two = rank_targets([[1., 0.]], vectors[::-1], ["c", "a", "b"], rows)
        self.assertEqual(one, two)
        self.assertEqual(one[0], [1])

    def test_duplicate_ids_and_misaligned_features_rejected(self):
        for ids in (["same", "same"], ["one"]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                rank_targets([[1., 0.]], [[1., 0.], [0., 1.]], ids,
                             [{"reference": "ref", "target": "one"}])

    def test_metrics_have_known_single_target_values(self):
        metrics = retrieval_metrics([1, 2, 10, 51])
        self.assertEqual(metrics["recall@1"], .25)
        self.assertEqual(metrics["recall@5"], .5)
        self.assertEqual(metrics["recall@10"], .75)
        self.assertEqual(metrics["recall@50"], .75)
        self.assertAlmostEqual(metrics["mrr"], (1 + .5 + .1 + 1/51)/4)

    def test_invalid_ranks_rejected(self):
        for ranks in ([], [0], [-1], [float("nan")], [1.5]):
            with self.subTest(ranks=ranks), self.assertRaises(ValueError):
                retrieval_metrics(ranks)

    def test_subset_order_is_deterministic_and_ignores_input_order(self):
        rows = [{"candidate": f"r{i}", "target": f"t{i}", "captions": ["first", "second"]} for i in range(12)]
        self.assertEqual(stable_order(rows, "dress", "train"), stable_order(rows[::-1], "dress", "train"))
        self.assertEqual(select_pairs(rows, "dress", "train", 4), select_pairs(rows[::-1], "dress", "train", 4))

    def test_internal_development_images_disjoint_from_training(self):
        rows = [{"candidate": f"r{i}", "target": f"t{i}", "captions": ["first", "second"]} for i in range(12)]
        train = select_pairs(rows, "dress", "train", 4)
        dev = select_pairs(rows, "dress", "dev", 3, forbidden_ids=image_ids(train))
        self.assertFalse(image_ids(train) & image_ids(dev))

    def test_pair_selection_rejects_self_reference_and_duplicate_pairs(self):
        row = {"candidate": "ref", "target": "target", "captions": ["first", "second"]}
        rows = [row, row.copy(), {"candidate": "same", "target": "same", "captions": ["first", "second"]}]
        self.assertEqual(len(select_pairs(rows, "dress", "train", 1)), 1)
        with self.assertRaises(ValueError):
            select_pairs(rows, "dress", "train", 2)

    def test_fusion_uses_only_shared_dimensional_clip_space(self):
        joined = fusion_inputs([[3., 4.]], [[4., 3.]])
        self.assertEqual(joined.shape, (1, 6))
        with self.assertRaises(ValueError):
            fusion_inputs([[1., 0.]], [[1., 0., 0.]])
        for bad in ([[0., 0.]], [[float("nan"), 0.]]):
            with self.assertRaises(ValueError):
                normalized(bad)

    def test_learned_fusion_starts_at_fixed_baseline_and_can_receive_gradients(self):
        import torch
        torch.manual_seed(42)
        model = build_fusion(dimension=4, hidden=3)
        reference = torch.nn.functional.normalize(torch.rand(5, 4), dim=-1)
        text = torch.nn.functional.normalize(torch.rand(5, 4), dim=-1)
        output = model(reference, text)
        torch.testing.assert_close(output, torch.nn.functional.normalize(reference+text, dim=-1))
        output[:, 0].sum().backward()
        self.assertGreater(float(model.layers[-1].weight.grad.norm()), 0.)

    def test_recorded_loader_rejects_reference_in_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "results/fashion"
            output.mkdir(parents=True)
            (output / "summary.json").write_text(json.dumps({"status": "complete"}))
            row = {"reference": "r", "target": "t", "methods": {method: {"target_rank": 1, "top_ids": ["t"]} for method in METHODS}}
            path = output / "queries.json"
            path.write_text(json.dumps([row]))
            self.assertEqual(load_fashion_results(root)["queries"][0]["target"], "t")
            row["methods"][METHODS[0]]["top_ids"] = ["r"]
            path.write_text(json.dumps([row]))
            with self.assertRaises(ValueError):
                load_fashion_results(root)

    def test_real_benchmark_pools_are_image_disjoint_and_metrics_reconstruct(self):
        measured = load_fashion_results(SOURCE_ROOT)
        summary = measured["summary"]
        manifest = json.loads((SOURCE_ROOT / "assets/benchmarks/fashion/manifest.json").read_text())
        pools = {phase: set(sum(categories.values(), [])) for phase, categories in manifest["candidate_pools"].items()}
        for first, second in (("train", "dev"), ("train", "evaluation"), ("dev", "evaluation")):
            self.assertFalse(pools[first] & pools[second])
            first_hashes = {manifest["images"][item]["sha256"] for item in pools[first]}
            second_hashes = {manifest["images"][item]["sha256"] for item in pools[second]}
            self.assertFalse(first_hashes & second_hashes)
        self.assertEqual(summary["phase_queries"]["evaluation"], len(measured["queries"]))
        for metric in summary["category_metrics"]:
            queries = [row for row in measured["queries"] if row["category"] == metric["category"]]
            reconstructed = retrieval_metrics([row["methods"][metric["method"]]["target_rank"] for row in queries])
            for key in ("recall@1", "recall@5", "recall@10", "recall@50", "mrr"):
                self.assertAlmostEqual(reconstructed[key], metric[key])
        curve = summary["training_curve"]
        self.assertEqual(summary["selected_epoch"], max(curve, key=lambda row: row["dev_macro_recall@10"])["epoch"])

    def test_hash_repair_protects_evaluation_then_development_without_replacement(self):
        images = {item: {"sha256": digest} for item, digest in {
            "train-a": "A", "train-b": "B", "train-x": "X", "dev-b": "B", "dev-c": "C",
            "dev-y": "Y", "eval-a": "A", "eval-c": "C", "eval-z": "Z"}.items()}
        def row(name, reference, target):
            return {"query_id": name, "reference": reference, "target": target}
        manifest = {"protocol": {}, "images": images,
                    "candidate_pools": {"train": {"dress": ["train-a", "train-b", "train-x"]},
                                        "dev": {"dress": ["dev-b", "dev-c", "dev-y"]},
                                        "evaluation": {"dress": ["eval-a", "eval-c", "eval-z"]}},
                    "queries": {"train": [row("t1", "train-a", "train-x"), row("t2", "train-b", "train-x")],
                                "dev": [row("d1", "dev-c", "dev-y")],
                                "evaluation": [row("e1", "eval-a", "eval-z")]}}
        fixed = enforce_image_hash_disjoint(manifest)
        self.assertEqual(fixed["queries"]["evaluation"], manifest["queries"]["evaluation"])
        self.assertEqual(fixed["candidate_pools"]["evaluation"], manifest["candidate_pools"]["evaluation"])
        self.assertEqual(fixed["candidate_pools"]["dev"]["dress"], ["dev-b", "dev-y"])
        self.assertEqual(fixed["candidate_pools"]["train"]["dress"], ["train-x"])
        self.assertEqual(fixed["data_integrity_amendment"]["initial_pairwise_overlap_count_sum"], 3)
        self.assertFalse(fixed["data_integrity_amendment"]["replacement_sampling"])
        self.assertEqual(enforce_image_hash_disjoint(fixed), fixed)

    def test_final_evaluation_and_frozen_baselines_preserved_after_integrity_amendment(self):
        manifest = json.loads((SOURCE_ROOT / "assets/benchmarks/fashion/manifest.json").read_text())
        amendment = manifest["data_integrity_amendment"]
        self.assertTrue(amendment["evaluation_queries_and_entire_gallery_preserved"])
        self.assertEqual(amendment["initial_pairwise_hash_overlap_counts"],
                         {"train_dev": 1, "train_evaluation": 3, "dev_evaluation": 1})
        self.assertEqual(amendment["removed"]["train"]["query_count"], 3)
        self.assertEqual(amendment["removed"]["dev"]["query_count"], 0)
        self.assertEqual(len(manifest["queries"]["evaluation"]), 287)

    def test_custom_request_uses_safe_checkpoint_and_never_assigns_target_metrics(self):
        import torch
        manifest = json.loads((SOURCE_ROOT / "assets/benchmarks/fashion/manifest.json").read_text())
        query = manifest["queries"]["evaluation"][0]
        with np.load(SOURCE_ROOT / "assets/benchmarks/fashion/features.npz", allow_pickle=False) as data:
            vector = data["image_features"][0].copy()
        _composition_assets.cache_clear()
        with mock.patch("shoplens.fashion_retrieval.encode_fashion_text", return_value=vector) as encode, \
                mock.patch("torch.load", wraps=torch.load) as load:
            result = compose_fashion_query(SOURCE_ROOT, Path("/unused-prepared-runtime"), query["reference"],
                                           "make the color brighter", query["category"], limit=5)
        encode.assert_called_once()
        self.assertTrue(load.call_args.kwargs["weights_only"])
        self.assertFalse(result["labels_available"])
        self.assertNotIn("target_rank", json.dumps(result))
        for record in result["methods"].values():
            self.assertEqual(len(record["top_ids"]), 5)
            self.assertEqual(len(record["scores"]), 5)
            self.assertNotIn(query["reference"], record["top_ids"])
            self.assertEqual(record["scores"], sorted(record["scores"], reverse=True))

    def test_invalid_custom_input_never_calls_text_encoder(self):
        with mock.patch("shoplens.fashion_retrieval.encode_fashion_text") as encode:
            for text, category, reference in (("", "dress", "any"), ("blue", "unknown", "any"),
                                               ("blue", "dress", "not-a-candidate")):
                with self.subTest(text=text, category=category), self.assertRaises(ValueError):
                    compose_fashion_query(SOURCE_ROOT, SOURCE_ROOT, reference, text, category)
            encode.assert_not_called()


if __name__ == "__main__":
    unittest.main()
