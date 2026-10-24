"""Isolated UI fixtures, not trained artifacts or scientific experiment results.

The tiny deterministic backends below test routing, rendering and session state.
They never write to the coursework artifacts directory and are not performance tests.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest


UI_PATH = Path(__file__).resolve().parents[1] / "src" / "shoplens" / "ui.py"
spec = importlib.util.spec_from_file_location("shoplens_qwen_ui_unit", UI_PATH)
ui = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(ui)


class InterfaceContractTests(unittest.TestCase):
    def test_model_registry_uses_independent_seed42_namespace(self):
        old = Path("/fixture/artifacts/baby")
        self.assertEqual(ui._model_source(old, "text"), ("shoplens.inference", old))
        self.assertEqual(ui._model_source(old, "qwen_semantic"), ("shoplens.qwen_inference", Path("/fixture/artifacts/qwen/baby/qwen_semantic/seed42")))
        with self.assertRaises(ValueError):
            ui._model_source(old, "unknown")

    def test_learned_qwen_hashes_can_vary_but_legacy_hashes_cannot(self):
        rows = [{"model": "qwen_semantic", "seed": seed, "items_sha256": "same", "selected_features_sha256": str(seed), "revision": "pinned"} for seed in (42, 43, 44)]
        self.assertTrue(ui._metric_compatibility(rows))
        rows[1]["revision"] = "different"
        self.assertFalse(ui._metric_compatibility(rows))
        for row in rows:
            row.update(model="adapted", revision="same")
        self.assertFalse(ui._metric_compatibility(rows))

    def test_duplicate_seeds_or_different_eval_protocol_are_not_merged(self):
        self.assertFalse(ui._metric_compatibility([{"model": "qwen_frozen", "seed": 42}] * 2))
        self.assertFalse(ui._metric_compatibility([{"model": "text", "seed": 42, "n_targets": 10}, {"model": "text", "seed": 43, "n_targets": 20}]))

    def test_qwen_comparison_checks_shared_base_but_not_device_or_candidate_encoder(self):
        rows = [{"model": model, "seed": 42, "featurehash": "same base", "training_configuration": {"model_id": "pinned", "revision": "same", "objective": model}, "run_protocol": {"candidate_encoder": model, "device": device}} for model, device in zip(ui.QWEN_MODES, ("cpu", "cuda", "cuda:0"))]
        self.assertTrue(ui._metric_compatibility(rows))
        rows[1]["featurehash"] = "different base"
        self.assertFalse(ui._metric_compatibility(rows))
        rows[1]["featurehash"] = "same base"
        rows[1]["training_configuration"]["revision"] = "different revision"
        self.assertFalse(ui._metric_compatibility(rows))

    def test_rank_changes_do_not_impute_outside_top_k(self):
        self.assertEqual(ui._rank_delta(None, 2), "Entered Top-K")
        self.assertEqual(ui._rank_delta(3, None), "Left Top-K")
        self.assertEqual(ui._rank_delta(4, 2), "↑ 2")

    def test_training_plot_only_uses_actual_finite_epoch_scalars(self):
        rows = ui._training_traces({"adaptation_epochs": [{"epoch": 1, "semantic_loss": .7}, {"epoch": 2, "semantic_loss": .5, "validation_ndcg@10": .02}, {"epoch": 3, "semantic_loss": float("nan")}], "missing": [1, 2]})
        self.assertEqual(len(rows), 3)
        self.assertEqual([row["value"] for row in rows if row["trace"].endswith("semantic_loss")], [.7, .5])

    def test_image_paths_cannot_escape_project_root(self):
        with tempfile.TemporaryDirectory(prefix="shoplens-ui-path-fixture-") as directory:
            root = Path(directory)
            self.assertIsNone(ui._local_image(root, {"image_path": "/etc/passwd"}))
            self.assertIsNone(ui._local_image(root, {"image_path": "../../outside.jpg"}))


class StreamlitFixtureTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="shoplens-ui-synthetic-fixture-")
        self.root = Path(self.temporary.name).resolve()
        self.calls = []
        self.processed = self.root / "data" / "processed" / "baby"
        self.processed.mkdir(parents=True)
        self.items = [{"item_id": f"FIXTURE-{i:02d}", "title": f"Synthetic UI fixture item {i}", "description": "Fixture only; never a real product or result."} for i in range(12)]
        (self.processed / "items.jsonl").write_text("\n".join(json.dumps(item) for item in self.items))
        (self.processed / "audit.json").write_text(json.dumps({"synthetic": True, "source": "isolated UI fixture, not experiment data"}))
        self.old = self.root / "artifacts" / "baby"
        (self.old / "models").mkdir(parents=True)
        for model in ("popularity", "mf", "tfidf", "text", "multimodal", "adapted"):
            (self.old / "models" / f"{model}.json").write_text("{}")
        self.modules = {name: self.backend(name) for name in ("shoplens.inference", "shoplens.qwen_inference")}
        self.module_patch = patch.dict(sys.modules, self.modules)
        self.module_patch.start()
        self.env_patch = patch.dict("os.environ", {"SHOPLENS_ROOT": str(self.root)})
        self.env_patch.start()

    def tearDown(self):
        self.env_patch.stop()
        self.module_patch.stop()
        self.temporary.cleanup()

    def backend(self, name):
        module = types.ModuleType(name)
        module.load_items = lambda directory: [json.loads(line) for line in (directory / "items.jsonl").read_text().splitlines()]
        def available(directory):
            if name.endswith("qwen_inference"):
                return [directory.parent.name] if (directory / "complete.json").is_file() else []
            return [file.stem for file in (directory / "models").glob("*.json")]
        def recommend(processed, directory, model, liked, k=10):
            self.calls.append((name, directory, model, list(liked)))
            candidates = [item for item in self.items if item["item_id"] not in liked]
            # Deterministic fixture behavior, explicitly marked synthetic in audit.
            if model == "qwen_semantic":
                candidates = candidates[1:] + candidates[:1]
            elif model == "qwen_preference":
                candidates = candidates[::-1]
            return [{"item_id": item["item_id"], "rank": i+1, "score": 1/(i+1)} for i, item in enumerate(candidates[:k])]
        module.available_models = available
        module.recommend = recommend
        module.explain = lambda processed, directory, model, liked, item_id: {"model": model, "note": "Synthetic isolated UI fixture; not scientific evidence.", "history_evidence": [{"item_id": item, "text_similarity": .1} for item in liked[-20:]]}
        return module

    def app(self, workspace="Research lab"):
        source = f'import importlib.util\nimport streamlit as st\nst.session_state.setdefault("shoplens_workspace", {workspace!r})\nspec = importlib.util.spec_from_file_location("fixture_shop_ui", {str(UI_PATH)!r})\nui = importlib.util.module_from_spec(spec)\nspec.loader.exec_module(ui)\nui.main()\n'
        return AppTest.from_string(source, default_timeout=30)

    def test_workspace_routes_to_studio_and_can_return_to_research(self):
        import streamlit as st
        calls = []
        studio = types.ModuleType("shoplens.studio_ui")
        def render(*args):
            calls.append(args)
            st.markdown("DIAGNOSTIC_STUDIO: isolated routing fixture")
        studio.render_studio = render
        with patch.dict(sys.modules, {"shoplens.studio_ui": studio}):
            app = self.app(workspace="Shopping Studio").run()
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(calls[-1][1], "baby")
            self.assertEqual(len(calls[-1][2]), len(self.items))
            self.assertTrue(any("DIAGNOSTIC_STUDIO" in element.value for element in app.markdown))
            next(widget for widget in app.radio if widget.label == "Workspace").set_value("Research lab").run()
            self.assertEqual(len(app.exception), 0)
            self.assertTrue(any(widget.label == "Recommendation method" for widget in app.selectbox))

    def ready_qwen(self):
        for model in ui.QWEN_MODES:
            directory = self.root / "artifacts" / "qwen" / "baby" / model / "seed42"
            directory.mkdir(parents=True)
            (directory / "complete.json").write_text(json.dumps({"fixture": True, "model": model}))

    def test_missing_qwen_preserves_six_legacy_choices_and_pending_status(self):
        app = self.app().run()
        self.assertEqual(len(app.exception), 0)
        selector = next(widget for widget in app.selectbox if widget.label == "Recommendation method")
        self.assertEqual(len(selector.options), 6)
        self.assertTrue(any("0/3" in alert.value for alert in app.info))
        self.assertTrue(any("Synthetic test artifacts" in alert.value for alert in app.warning))

    def test_three_ready_modes_use_qwen_backend_and_shared_complete_history(self):
        self.ready_qwen()
        app = self.app().run()
        app.session_state["shoplens_likes_baby"] = [item["item_id"] for item in self.items[:3]]
        app.run()
        self.assertEqual(len(app.exception), 0)
        selector = next(widget for widget in app.multiselect if widget.label == "Models to compare")
        self.assertEqual(selector.value, list(ui.QWEN_MODES))
        self.assertTrue(self.calls)
        for backend, directory, model, history in self.calls:
            self.assertEqual(history, [item["item_id"] for item in self.items[:3]])
            if model in ui.QWEN_MODES:
                self.assertEqual(backend, "shoplens.qwen_inference")
                self.assertEqual(directory, self.root / "artifacts" / "qwen" / "baby" / model / "seed42")
        self.assertTrue(any("Full rank changes" in element.value for element in app.markdown))
        # Removing a thumbnail updates the single shared preference list.
        button = next(button for button in app.button if str(button.key).startswith("history_remove_catalog_"))
        button.click().run()
        self.assertEqual(len(app.session_state["shoplens_likes_baby"]), 2)
        self.assertEqual(len(app.exception), 0)

    def test_empty_root_is_honest_and_does_not_infer(self):
        (self.processed / "items.jsonl").unlink()
        app = self.app().run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(self.calls, [])
        self.assertTrue(any("No recommendation results or performance metrics are available yet" in element.value for element in app.markdown))

    def test_new_ready_artifacts_change_default_comparison_after_old_session(self):
        app = self.app().run()
        app.session_state["shoplens_likes_baby"] = [item["item_id"] for item in self.items[:3]]
        app.run()
        self.ready_qwen()
        app.run()
        self.assertEqual(len(app.exception), 0)
        selector = next(widget for widget in app.multiselect if widget.label == "Models to compare")
        self.assertEqual(selector.value, list(ui.QWEN_MODES))

    def test_single_seed_result_has_no_fabricated_standard_deviation(self):
        directory = self.root / "artifacts" / "qwen" / "baby"
        directory.mkdir(parents=True)
        records = [{"model": "qwen_frozen", "seed": 42, "split": "test", "ndcg@10": .01, "items_sha256": "fixture only", "selected_features_sha256": "fixture only", "run_protocol": {"evaluation_subset": False}}]
        (directory / "results.json").write_text(json.dumps({"synthetic": True, "results": records}))
        app = self.app().run()
        self.assertEqual(len(app.exception), 0)
        summaries = [frame.value for frame in app.dataframe if "score" in frame.value.columns]
        self.assertEqual(len(summaries), 1)
        self.assertEqual(summaries[0].iloc[0]["score"], "0.01000 (n=1)")
        self.assertNotIn("±", summaries[0].iloc[0]["score"])

    def test_qwen_metric_colors_order_and_sample_sd_preserve_legacy_families(self):
        directory = self.root / "artifacts" / "qwen" / "baby"
        directory.mkdir(parents=True)
        def records(models):
            return [{"model": model, "seed": seed, "split": "test", "ndcg@10": value,
                     "items_sha256": "fixture only", "run_protocol": {"evaluation_subset": False}}
                    for model in models for seed, value in ((42, .01), (43, .03))]
        (directory / "results.json").write_text(json.dumps({"synthetic": True, "results": records(reversed(ui.QWEN_MODES))}))
        legacy = [model for model in ui.MODELS if model not in ui.QWEN_MODES]
        (self.old / "results.json").write_text(json.dumps({"synthetic": True, "results": records(legacy)}))
        app = self.app().run()
        self.assertEqual(len(app.exception), 0)
        traces = json.loads(app.get("plotly_chart")[0].proto.spec)["data"]
        self.assertEqual([trace["name"] for trace in traces], [ui.MODEL_SHORT[model] for model in ui.QWEN_MODES])
        self.assertEqual([trace["marker"]["color"].lower() for trace in traces], ["#546982", "#347cdb", "#1a967b"])
        summary = next(frame.value for frame in app.dataframe if "score" in frame.value.columns)
        self.assertEqual(list(summary["method"]), [ui.MODEL_SHORT[model] for model in ui.QWEN_MODES])
        self.assertEqual(list(summary["score"]), ["0.02000 ± 0.01414 (n=2)"] * 3)
        for family, expected in (("Six original baselines", legacy), ("All methods", legacy + list(ui.QWEN_MODES))):
            next(widget for widget in app.radio if widget.label == "Experiment family").set_value(family).run()
            self.assertEqual(len(app.exception), 0)
            traces = json.loads(app.get("plotly_chart")[0].proto.spec)["data"]
            self.assertEqual({trace["name"] for trace in traces}, {ui.MODEL_SHORT[model] for model in expected})


if __name__ == "__main__":
    unittest.main()
