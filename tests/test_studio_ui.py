"""Synthetic Studio interaction fixtures; no model or real experiment execution."""
from __future__ import annotations

import importlib.util
import io
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image
from streamlit.testing.v1 import AppTest

SOURCE = Path(__file__).resolve().parents[1] / "src/shoplens"


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, SOURCE / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


MODULE_NAME = "shoplens.studio_ui_diagnostic_fixture"
ui = load_module(MODULE_NAME, "studio_ui.py")
backend = load_module("shoplens.studio_backend_diagnostic_fixture", "studio.py")
ITEMS = [{"item_id": f"D{i}", "title": f"DIAGNOSTIC synthetic yoga product {i}",
          "description": f"DIAGNOSTIC catalog description {i}; not actual product evidence.",
          "metadata": {"brand": "DIAGNOSTIC brand", "categories": [["Diagnostic", "Yoga"]],
                       **({"price": float(10 + i * 5)} if i != 5 else {})}}
         for i in range(6)]
CATALOG = {item["item_id"]: item for item in ITEMS}
TEXT = np.asarray([[1., 0., .1], [.9, .2, .1], [.8, .3, .2], [.2, .9, .2], [.1, .2, 1.], [.4, .8, .5]])
IMAGES = np.asarray([[1., .1], [.8, .2], [.6, .4], [.4, .6], [.2, .8], [.1, 1.]])


class StudioPureBoundaryTests(unittest.TestCase):
    def test_image_validation_checks_actual_bytes_and_format(self):
        image = io.BytesIO()
        Image.new("RGB", (8, 8), "blue").save(image, format="PNG")
        self.assertEqual(ui._validated_image(image.getvalue()), image.getvalue())
        for payload in (b"", b"not an image", b"x" * (ui.MAX_IMAGE_BYTES + 1)):
            with self.subTest(size=len(payload)), self.assertRaises(ValueError):
                ui._validated_image(payload)
        image = io.BytesIO()
        Image.new("RGB", (8, 8), "blue").save(image, format="GIF")
        with self.assertRaises(ValueError):
            ui._validated_image(image.getvalue())

    def test_feedback_is_exclusive_and_never_mutates_shared_likes(self):
        shared = ["D0"]
        state = {"positive": list(shared), "negative": [], "hidden": []}
        ui._feedback(state, "D0", "negative")
        self.assertEqual(state, {"positive": [], "negative": ["D0"], "hidden": []})
        ui._feedback(state, "D0", "hidden")
        self.assertEqual(state, {"positive": [], "negative": [], "hidden": ["D0"]})
        ui._feedback(state, "D0", None)
        self.assertEqual(state, {"positive": [], "negative": [], "hidden": []})
        self.assertEqual(shared, ["D0"])

    def test_fingerprint_tracks_every_search_input(self):
        original = ["yoga", "pink", 50, b"diagnostic bytes", .65, ("fixture", 1)]
        token = ui._fingerprint(*original)
        self.assertEqual(token, ui._fingerprint(*original))
        self.assertNotEqual(token, ui._fingerprint(*original, required_category="Diagnostic / Yoga"))
        replacements = ("shoes", "red", 40, b"other bytes", .3, ("fixture", 2))
        for index, replacement in enumerate(replacements):
            values = list(original)
            values[index] = replacement
            with self.subTest(index=index):
                self.assertNotEqual(token, ui._fingerprint(*values))

    def test_html_escapes_source_content_and_unknown_price(self):
        malicious = {"item_id": "<script>alert(1)</script>", "title": '<img src=x onerror="alert(1)">',
                     "description": "<script>bad()</script>",
                     "metadata": {"price": "$10 - $20", "brand": "<b>unsafe</b>"}}
        output = ui._result_html("/DIAGNOSTIC_ONLY", [malicious],
                                 {"query": "<script>query()</script>", "avoid": "<svg>"},
                                 malicious["item_id"])
        self.assertNotIn("<script>", output)
        self.assertNotIn("<img src=x", output)
        self.assertIn("&lt;script&gt;", output)
        self.assertIn("Unknown", output)
        self.assertIn("historical USD metadata", output)
        self.assertIsNone(ui._price(malicious))
        self.assertEqual(ui._metadata(ITEMS[0])["Historical catalog price (USD)"], "$10.00")

    def test_rank_changes_only_use_observed_positions(self):
        before = [{"item_id": "D0", "rank": 1}, {"item_id": "D1", "rank": 2}]
        after = [{"item_id": "D1", "rank": 1}, {"item_id": "D2", "rank": 2}]
        change = ui._rank_changes(before, after)
        self.assertEqual(change["shared"], 1)
        self.assertEqual(change["entered"], ["D2"])
        self.assertEqual(change["exited"], ["D0"])
        self.assertEqual(change["moved"], 1)
        self.assertAlmostEqual(change["jaccard"], 1 / 3)
        self.assertNotIn("D2", change["before"])
        self.assertIsNone(ui._rank_changes([], [])["jaccard"])
        self.assertEqual(ui._rank_changes(before, before)["moved"], 0)


class StudioInteractionTests(unittest.TestCase):
    def setUp(self):
        self.calls = {"query": [], "image": [], "catalog": [], "model": [], "candidate": []}
        self.model_revision, self.failed_model = 1, None
        self.text_vectors, self.image_vectors = TEXT, IMAGES
        self.candidate_text_by_model = {}
        self.encoder = SimpleNamespace(
            status=lambda root: {"ready": True, "query_ready": True, "image_ready": True},
            load_catalog=self.catalog, encode_query=self.query, encode_queries=self.queries,
            query_encoding_metadata=self.encoding_metadata, encode_image=self.image)
        self.patches = [patch.dict(sys.modules, {MODULE_NAME: ui}),
                        patch.object(ui, "_encoder_api", return_value=self.encoder),
                        patch.object(ui, "_model_api", return_value=SimpleNamespace(model_status=self.model_status, personalization=self.personalization, candidate_catalog=self.candidate_catalog)),
                        patch.object(ui, "_studio_api", return_value=backend)]
        ui._catalog_vectors.clear()
        for context in self.patches:
            context.start()
        self.addCleanup(self.cleanup)

    def cleanup(self):
        for context in reversed(self.patches):
            context.stop()
        ui._catalog_vectors.clear()

    def catalog(self, root, domain):
        self.calls["catalog"].append(domain)
        return {"items": ITEMS, "item_ids": list(CATALOG), "text": self.text_vectors, "image": self.image_vectors,
                "signature": ("DIAGNOSTIC", 1), "metadata": {"diagnostic": True}}

    def model_status(self, root, domain):
        return {mode: {"ready": True, "seed": 42, "artifact_signature": f"DIAGNOSTIC-{domain}-{mode}-{self.model_revision}"} for mode in ui.MODES}

    def encoding_metadata(self, root, domain):
        return {mode: {"model": mode, "model_id": "DIAGNOSTIC-ENCODER", "adapter_sha256": None if mode == ui.MODES[0] else f"DIAGNOSTIC-ADAPTER-{mode}",
                       "artifact_signature": self.model_status(root, domain)[mode]["artifact_signature"],
                       "encoding_protocol": "DIAGNOSTIC synthetic protocol, not actual model output"} for mode in ui.MODES}

    def candidate_catalog(self, root, domain, mode):
        self.calls["candidate"].append((domain, mode))
        return {"item_ids": list(CATALOG), "text": self.candidate_text_by_model.get(mode, self.text_vectors),
                "metadata": {"model": mode, "candidate_sha256": f"DIAGNOSTIC-CANDIDATE-{mode}",
                             "artifact_signature": self.model_status(root, domain)[mode]["artifact_signature"]}}

    def personalization(self, root, domain, mode, positive):
        self.calls["model"].append((domain, mode, tuple(positive)))
        if mode == self.failed_model:
            raise RuntimeError("DIAGNOSTIC unavailable selected head")
        values = {"qwen_frozen": [0, 1, 2, 3, 4, 5], "qwen_semantic": [5, 4, 3, 2, 1, 0],
                  "qwen_preference": [0, 5, 1, 4, 2, 3]}
        return {"scores": np.asarray(values[mode], dtype=float) if positive else None,
                "metadata": {"model": mode, "label": ui.MODEL_LABELS[mode], "seed": 42,
                             "selected_epoch": 1, "active": bool(positive), "history_count": len(positive),
                             "head_sha256": "DIAGNOSTIC-HASH-NOT-FORMAL", "artifact_signature": self.model_status(root, domain)[mode]["artifact_signature"]}}

    def query(self, root, text):
        self.calls["query"].append(text)
        return np.asarray([1., 0., .1])

    def queries(self, root, domain, text):
        self.calls["query"].append(text)
        return {mode: np.asarray(vector) for mode, vector in zip(ui.MODES, ([1., 0., .1], [0., 1., .1], [.1, 0., 1.]))}

    def image(self, root, payload):
        self.calls["image"].append(payload)
        return np.asarray([1., .1])

    def app(self, liked=()):
        source = f'''
import sys
import streamlit as st
ui = sys.modules[{MODULE_NAME!r}]
st.warning("DIAGNOSTIC synthetic Studio fixture; not actual recommendations or formal evidence.")
items = {ITEMS!r}
catalog = {{item["item_id"]: item for item in items}}
domain = st.selectbox("DIAGNOSTIC collection", ["sports", "baby"])
ui.render_studio("/DIAGNOSTIC_ONLY_NO_FILES", domain, items, catalog, [], {list(liked)!r}, ("DIAGNOSTIC", domain, 1))
'''
        return AppTest.from_string(source, default_timeout=30).run()

    def clean(self, app):
        self.assertEqual(len(app.exception), 0, [exception.value for exception in app.exception])
        self.assertEqual(len(app.error), 0, [error.value for error in app.error])
        self.assertTrue(any("DIAGNOSTIC synthetic" in item.value for item in app.warning))

    @staticmethod
    def button(app, label):
        return next(item for item in app.button if item.label == label)

    @staticmethod
    def input(app, label):
        return next(item for item in app.text_input if item.label == label)

    def search(self, app):
        self.input(app, "Search request").set_value("yoga mat").run()
        self.button(app, "Find products").click().run()
        self.clean(app)
        return app

    def test_first_page_is_lazy_and_brief_rules_are_editable(self):
        app = self.app()
        self.clean(app)
        self.assertEqual(self.calls["query"], [])
        self.assertEqual(self.calls["catalog"], [])
        self.assertTrue(self.button(app, "Find products").disabled)
        app.text_area[0].set_value("Find yoga mats under $25; avoid pink")
        self.button(app, "Interpret brief").click().run()
        self.clean(app)
        self.assertIn("yoga", self.input(app, "Search request").value)
        self.assertEqual(self.input(app, "Avoid (soft preference)").value, "pink")
        self.assertEqual(app.number_input[0].value, 25.)
        self.assertEqual(self.calls["query"], [])
        self.assertEqual(self.calls["catalog"], [])
        self.assertTrue(any("transparent keyword and budget rules" in item.value for item in app.caption))

    def test_feedback_reranks_without_reencoding(self):
        app = self.search(self.app(liked=("D0",)))
        self.assertEqual(self.calls["query"], ["yoga mat"])
        self.assertEqual(len(self.calls["model"]), 3)
        target = app.session_state["shoplens_studio_sports"]["rows"][0]["item_id"]
        next(button for button in app.button if button.key == f"studio_more_sports_{target}").click().run()
        self.clean(app)
        state = app.session_state["shoplens_studio_sports"]
        self.assertIn(target, state["positive"])
        self.assertTrue({"D0", target}.isdisjoint(row["item_id"] for row in state["rows"]))
        self.assertEqual(self.calls["query"], ["yoga mat"])
        self.assertEqual(self.calls["image"], [])
        self.assertEqual(len(self.calls["model"]), 6)
        self.assertTrue(all(target in call[2] for call in self.calls["model"][-3:]))
        self.assertTrue(any(button.label == "Shortlist reference" for button in app.button))

    def test_changed_request_hides_previous_results_until_search(self):
        app = self.search(self.app())
        self.input(app, "Search request").set_value("different request").run()
        self.clean(app)
        self.assertFalse(any(button.label == "More like this" for button in app.button))
        self.assertTrue(any("previous results are hidden" in info.value for info in app.info))
        self.assertEqual(self.calls["query"], ["yoga mat"])
        self.button(app, "Find products").click().run()
        self.clean(app)
        self.assertEqual(self.calls["query"], ["yoga mat", "different request"])

    def test_shortlist_compare_final_choice_and_downloads(self):
        app = self.search(self.app())
        first, second = [row["item_id"] for row in app.session_state["shoplens_studio_sports"]["rows"][:2]]
        for item_id in (first, second):
            next(button for button in app.button if button.key == f"studio_shortlist_sports_{item_id}").click().run()
            self.clean(app)
        self.assertEqual(app.session_state["shoplens_studio_sports"]["shortlist"], [first, second])
        self.assertTrue(any(item.label == "Products to compare" for item in app.multiselect))
        next(item for item in app.selectbox if item.label == "Final choice").set_value(first).run()
        self.button(app, "Confirm final choice").click().run()
        self.clean(app)
        self.assertEqual(app.session_state["shoplens_studio_sports"]["final_choice"], first)
        labels = {item.proto.label for item in app.get("download_button")}
        self.assertTrue({"Download shortlist JSON", "Download shortlist CSV", "Save result card (HTML)"}.issubset(labels))
        self.assertEqual(self.calls["query"], ["yoga mat"])

    def test_domains_are_isolated_and_missing_encoder_has_no_fallback(self):
        app = self.search(self.app())
        next(item for item in app.selectbox if item.label == "DIAGNOSTIC collection").set_value("baby").run()
        self.clean(app)
        self.assertEqual(app.session_state["shoplens_studio_baby"]["rows"], [])
        self.assertEqual(self.input(app, "Search request").value, "")
        self.encoder.status = lambda root: {"query_ready": False, "image_ready": True,
                                             "qwen": {"error": "DIAGNOSTIC missing local encoder"}}
        self.input(app, "Search request").set_value("stroller").run()
        self.button(app, "Find products").click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(any("DIAGNOSTIC missing local encoder" in error.value for error in app.error))
        self.assertEqual(app.session_state["shoplens_studio_baby"]["rows"], [])
        self.assertEqual(self.calls["query"], ["yoga mat"])

    def test_returning_to_collection_restores_request_results_and_final_choice(self):
        app = self.search(self.app())
        first = app.session_state["shoplens_studio_sports"]["rows"][0]["item_id"]
        next(button for button in app.button if button.key == f"studio_shortlist_sports_{first}").click().run()
        next(item for item in app.selectbox if item.label == "Final choice").set_value(first).run()
        self.button(app, "Confirm final choice").click().run()
        next(item for item in app.selectbox if item.label == "DIAGNOSTIC collection").set_value("baby").run()
        next(item for item in app.selectbox if item.label == "DIAGNOSTIC collection").set_value("sports").run()
        self.clean(app)
        self.assertEqual(self.input(app, "Search request").value, "yoga mat")
        self.assertEqual(next(item for item in app.selectbox if item.label == "Final choice").value, first)
        self.assertTrue(any(button.label == "More like this" for button in app.button))
        self.assertFalse(any("previous results are hidden" in info.value for info in app.info))
        self.assertEqual(self.calls["query"], ["yoga mat"])

    def test_no_history_uses_each_query_encoder_and_switch_reuses_encodings(self):
        app = self.search(self.app())
        groups = app.session_state["shoplens_studio_sports"]["rows_by_model"]
        self.assertNotEqual([row["item_id"] for row in groups[ui.MODES[0]]], [row["item_id"] for row in groups[ui.MODES[1]]])
        self.assertEqual(set(app.session_state["shoplens_studio_sports"]["query_vectors_by_model"]), set(ui.MODES))
        self.assertEqual({mode for _, mode in self.calls["candidate"]}, set(ui.MODES))
        self.assertTrue(any("Request retrieval is active" in item.value for item in app.caption))
        initial_calls = len(self.calls["model"])
        next(item for item in app.radio if item.label == "Learned recommendation objective").set_value(ui.MODES[1]).run()
        self.clean(app)
        self.assertEqual(self.calls["query"], ["yoga mat"])
        self.assertEqual(len(self.calls["model"]), initial_calls)
        state = app.session_state["shoplens_studio_sports"]
        self.assertEqual(state["selected_model"], ui.MODES[1])
        self.assertEqual(state["positive"], [])
        self.assertEqual(state["rank_reference_label"], "vs. Frozen Qwen")

    def test_switch_with_history_changes_rank_and_retains_the_decision(self):
        self.text_vectors = np.tile([1., 0., .1], (6, 1))
        self.image_vectors = np.tile([1., .1], (6, 1))
        app = self.search(self.app(liked=("D0",)))
        state = app.session_state["shoplens_studio_sports"]
        self.assertNotEqual([r["item_id"] for r in state["rows_by_model"][ui.MODES[0]]],
                            [r["item_id"] for r in state["rows_by_model"][ui.MODES[1]]])
        chosen = state["rows"][0]["item_id"]
        next(button for button in app.button if button.key == f"studio_shortlist_sports_{chosen}").click().run()
        next(item for item in app.selectbox if item.label == "Final choice").set_value(chosen).run()
        self.button(app, "Confirm final choice").click().run()
        next(item for item in app.radio if item.label == "Learned recommendation objective").set_value(ui.MODES[1]).run()
        self.clean(app)
        state = app.session_state["shoplens_studio_sports"]
        self.assertEqual(state["positive"], ["D0"])
        self.assertEqual(state["shortlist"], [chosen])
        self.assertEqual(state["final_choice"], chosen)
        self.assertEqual(self.input(app, "Search request").value, "yoga mat")
        self.assertEqual(self.calls["query"], ["yoga mat"])
        next(item for item in app.selectbox if item.label == "DIAGNOSTIC collection").set_value("baby").run()
        next(item for item in app.selectbox if item.label == "DIAGNOSTIC collection").set_value("sports").run()
        self.clean(app)
        self.assertEqual(next(item for item in app.radio if item.label == "Learned recommendation objective").value, ui.MODES[1])
        self.assertEqual(app.session_state["shoplens_studio_sports"]["final_choice"], chosen)
        self.assertEqual(self.calls["query"], ["yoga mat"])

    def test_stale_request_and_replaced_adapter_require_new_search(self):
        app = self.search(self.app(liked=("D0",)))
        initial = len(self.calls["model"])
        self.model_revision += 1
        app.run()
        self.clean(app)
        self.assertEqual(len(self.calls["model"]), initial)
        self.assertEqual(self.calls["query"], ["yoga mat"])
        self.assertTrue(any("query-encoding policy changed" in item.value for item in app.info))
        self.button(app, "Find products").click().run()
        self.clean(app)
        self.assertEqual(len(self.calls["model"]), initial + 3)
        self.assertEqual(self.calls["query"], ["yoga mat", "yoga mat"])
        self.input(app, "Search request").set_value("new pending request").run()
        next(item for item in app.radio if item.label == "Learned recommendation objective").set_value(ui.MODES[2]).run()
        self.clean(app)
        self.assertEqual(len(self.calls["model"]), initial + 3)
        self.assertFalse(any(button.label == "More like this" for button in app.button))
        self.assertTrue(any("Previous model rows are hidden" in item.value for item in app.info))

    def test_failed_group_clears_its_previous_rows_without_fallback(self):
        app = self.search(self.app(liked=("D0",)))
        self.failed_model = ui.MODES[1]
        self.model_revision += 1
        self.button(app, "Find products").click().run()
        next(item for item in app.radio if item.label == "Learned recommendation objective").set_value(ui.MODES[1]).run()
        self.assertEqual(len(app.exception), 0)
        state = app.session_state["shoplens_studio_sports"]
        self.assertEqual(state["rows_by_model"][ui.MODES[1]], [])
        self.assertTrue(state["rows_by_model"][ui.MODES[0]])
        self.assertTrue(any("DIAGNOSTIC unavailable selected head" in item.value for item in app.error))
        self.assertEqual(self.calls["query"], ["yoga mat", "yoga mat"])

    def test_session_reference_survives_model_switch_without_reencoding(self):
        app = self.app()
        image = io.BytesIO()
        Image.new("RGB", (8, 8), "blue").save(image, format="PNG")
        payload = image.getvalue()
        state = app.session_state["shoplens_studio_sports"]
        state["uploaded_reference_bytes"] = payload
        state["retained_reference_active"] = True
        app.session_state["studio_image_source_sports"] = "Upload image"
        app = self.search(app)
        self.assertEqual(self.calls["image"], [payload])
        next(item for item in app.radio if item.label == "Learned recommendation objective").set_value(ui.MODES[2]).run()
        self.clean(app)
        self.assertEqual(app.session_state["shoplens_studio_sports"]["uploaded_reference_bytes"], payload)
        self.assertEqual(self.calls["image"], [payload])
        self.assertEqual(self.calls["query"], ["yoga mat"])

    def test_stale_export_keeps_saved_choices_but_omits_ranking_rows(self):
        with patch.object(backend, "shortlist_json", wraps=backend.shortlist_json) as exporter:
            app = self.search(self.app(liked=("D0",)))
            item_id = app.session_state["shoplens_studio_sports"]["rows"][0]["item_id"]
            next(button for button in app.button if button.key == f"studio_shortlist_sports_{item_id}").click().run()
            self.clean(app)
            self.assertTrue(exporter.call_args.kwargs["recommendations"])
            self.input(app, "Search request").set_value("unsearched new need").run()
            self.clean(app)
            self.assertEqual(exporter.call_args.kwargs["recommendations"], [])
            self.assertFalse(exporter.call_args.args[2]["ranking_current"])
            self.assertIsNone(exporter.call_args.kwargs["source"]["selected_seed42_model"])
            self.assertEqual(app.session_state["shoplens_studio_sports"]["shortlist"], [item_id])
            self.assertEqual(exporter.call_args.args[2]["selected_model"], ui.MODES[0])

    def test_query_and_avoid_use_each_models_own_candidate_space(self):
        self.candidate_text_by_model = {mode: np.roll(TEXT, index, axis=1) for index, mode in enumerate(ui.MODES)}
        app = self.app()
        self.input(app, "Avoid (soft preference)").set_value("pink").run()
        with patch.object(backend, "rank_products", wraps=backend.rank_products) as ranker:
            app = self.search(app)
            self.assertEqual(ranker.call_count, 3)
            state = app.session_state["shoplens_studio_sports"]
            for mode, call in zip(ui.MODES, ranker.call_args_list):
                np.testing.assert_array_equal(call.args[1], self.candidate_text_by_model[mode])
                np.testing.assert_array_equal(call.kwargs["query_vector"], state["query_vectors_by_model"][mode])
                np.testing.assert_array_equal(call.kwargs["avoid_vector"], state["avoid_vectors_by_model"][mode])
                self.assertEqual(state["model_metadata"][mode]["request_retrieval"]["query_encoding"]["model"], mode)
        self.assertEqual(self.calls["query"], ["yoga mat", "pink"])

    def test_old_base_only_state_requires_explicit_new_search(self):
        app = self.search(self.app())
        state = app.session_state["shoplens_studio_sports"]
        state["encoding_policy"] = "shared-base-retrieval-three-user-heads-v1"
        state.pop("query_vectors_by_model")
        app.run()
        self.clean(app)
        self.assertFalse(any(button.label == "More like this" for button in app.button))
        self.assertEqual(self.calls["query"], ["yoga mat"])
        self.button(app, "Find products").click().run()
        self.clean(app)
        self.assertEqual(self.calls["query"], ["yoga mat", "yoga mat"])
        self.assertEqual(app.session_state["shoplens_studio_sports"]["encoding_policy"], ui.RANKING_POLICY)

    def test_encoder_identity_change_during_encoding_never_displays_mixed_results(self):
        calls = []
        def moving_identity(root, domain):
            calls.append(1)
            if len(calls) == 2:
                self.model_revision += 1
            return self.encoding_metadata(root, domain)
        self.encoder.query_encoding_metadata = moving_identity
        app = self.app()
        self.input(app, "Search request").set_value("yoga mat").run()
        self.button(app, "Find products").click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(any("identity changed during request encoding" in item.value for item in app.error))
        self.assertFalse(any(button.label == "More like this" for button in app.button))
        self.assertEqual(app.session_state["shoplens_studio_sports"]["rows"], [])

    def test_image_only_and_feedback_only_do_not_encode_empty_text(self):
        app = self.app()
        image = io.BytesIO()
        Image.new("RGB", (8, 8), "blue").save(image, format="PNG")
        payload = image.getvalue()
        state = app.session_state["shoplens_studio_sports"]
        state["uploaded_reference_bytes"], state["retained_reference_active"] = payload, True
        app.session_state["studio_image_source_sports"] = "Upload image"
        self.encoder.status = lambda root: {"query_ready": False, "image_ready": True}
        app.run()
        self.button(app, "Find products").click().run()
        self.clean(app)
        self.assertEqual(self.calls["query"], [])
        self.assertEqual(self.calls["image"], [payload])
        self.assertTrue(all(app.session_state["shoplens_studio_sports"]["rows_by_model"].values()))
        self.assertTrue(any("image-only request" in item.value for item in app.info))
        next(item for item in app.selectbox if item.label == "DIAGNOSTIC collection").set_value("baby").run()
        app.session_state["shoplens_studio_baby"]["positive"] = ["D0"]
        app.run()
        self.button(app, "Find products").click().run()
        self.clean(app)
        self.assertEqual(self.calls["query"], [])
        self.assertTrue(all(app.session_state["shoplens_studio_baby"]["rows_by_model"].values()))


if __name__ == "__main__":
    unittest.main()

