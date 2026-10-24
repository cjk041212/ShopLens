"""Diagnostic synthetic UI fixtures, never training or formal result artifacts.

These tiny data frames test click boundaries and session invalidation. They do
not read actual feature caches, invoke a model, or measure recommendation quality.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from streamlit.testing.v1 import AppTest


MODULE_NAME = "shoplens.visual_ui_diagnostic_fixture"
MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "shoplens" / "visual_ui.py"
spec = importlib.util.spec_from_file_location(MODULE_NAME, MODULE_PATH)
visual = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(visual)

MODES = ("qwen_frozen", "qwen_semantic", "qwen_preference")
LABELS = dict(zip(MODES, ("Frozen fixture", "Semantic fixture", "Preference fixture")))
COLORS = dict(zip(MODES, ("#546982", "#347CDB", "#1A967B")))


class VisualPageInteractionTests(unittest.TestCase):
    def setUp(self):
        self.calls = {"map": [], "relations": [], "removals": [], "inspect": []}
        self.patches = [
            patch.dict(sys.modules, {MODULE_NAME: visual}),
            patch.object(visual, "_ready", return_value=list(MODES)),
            patch.object(visual, "_map", side_effect=self.map_fixture),
            patch.object(visual, "_relations", side_effect=self.relation_fixture),
            patch.object(visual, "_removals", side_effect=self.removal_fixture),
        ]
        for context in self.patches:
            context.start()
        self.addCleanup(self.stop_patches)

    def stop_patches(self):
        for context in reversed(self.patches):
            context.stop()

    @staticmethod
    def item(item_id):
        return {"item_id": item_id, "title": f"DIAGNOSTIC synthetic product {item_id}"}

    def map_fixture(self, root, domain, mode, liked, k, sample_size, signature):
        self.calls["map"].append((mode, tuple(liked), k, sample_size, signature))
        rows = []
        for index, item_id in enumerate([*liked, "C1", "C2", "B1"]):
            role = "history" if item_id in liked else "recommended" if item_id.startswith("C") and liked else "background"
            rank = int(item_id[-1]) if role == "recommended" else None
            rows.append({**self.item(item_id), "x": index * .1, "y": index * -.1,
                         "role": role, "active_history": item_id in liked,
                         "rank": rank, "score": .1 / rank if rank else None})
        points = pd.DataFrame(rows)
        return {"diagnostic": True, "synthetic": True, "points": points,
                "top_items": points[points["role"] == "recommended"],
                "projection": {"explained_variance_ratio": [.4, .25]},
                "catalog_size": len(points), "background_sample_size": 1}

    def relation_fixture(self, root, domain, mode, liked, k, signature):
        self.calls["relations"].append((mode, tuple(liked), k, signature))
        history = pd.DataFrame([self.item(item_id) for item_id in liked])
        candidates = pd.DataFrame([self.item(item_id) for item_id in ("C1", "C2")])
        matrix = pd.DataFrame([[.2, -.1]] * len(liked), index=list(liked), columns=["C1", "C2"])
        return {"diagnostic": True, "synthetic": True,
                "history": history, "candidates": candidates, "matrix": matrix}

    def removal_fixture(self, root, domain, mode, liked, k, signature):
        self.calls["removals"].append((mode, tuple(liked), k, signature))
        self.assertGreater(len(liked), 1, "UI must not request unsupported empty-history removal")
        summary = pd.DataFrame([{"removed_item_id": item_id, "title": self.item(item_id)["title"],
                                 "topk_overlap_ratio": .5, "diagnostic": True} for item_id in liked])
        effects = pd.DataFrame([{"removed_item_id": item_id, "target_item_id": candidate,
                                 "score_delta": .1 if candidate == "C1" else -.05,
                                 "diagnostic": True} for item_id in liked for candidate in ("C1", "C2")])
        return {"diagnostic": True, "synthetic": True, "available": True,
                "summary": summary, "effects": effects}

    def app(self, page, liked=()):
        source = f'''
import sys
import streamlit as st
visual = sys.modules[{MODULE_NAME!r}]
st.warning("DIAGNOSTIC synthetic UI fixture; not formal recommendation or experiment results.")
liked = st.session_state.get("diagnostic_likes", {list(liked)!r})
def inspect(item_id, key):
    st.markdown("DIAGNOSTIC_INSPECTOR: " + item_id)
kwargs = dict(root="/DIAGNOSTIC_ONLY_NO_FILES", domain="baby", models={list(MODES)!r},
              liked=liked, signature=("diagnostic fixture", 1), labels={LABELS!r}, colors={COLORS!r})
if {page!r} == "atlas":
    visual.atlas(**kwargs, render_product=inspect)
else:
    visual.preference_lab(**kwargs)
'''
        return AppTest.from_string(source, default_timeout=30).run()

    def assert_clean(self, app):
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.error), 0, [alert.value for alert in app.error])
        self.assertTrue(any("DIAGNOSTIC synthetic" in alert.value for alert in app.warning))

    @staticmethod
    def button(app, label):
        return next(button for button in app.button if button.label == label)

    @staticmethod
    def selectbox(app, label):
        return next(widget for widget in app.selectbox if widget.label == label)

    @staticmethod
    def removal_charts(app):
        return [chart for chart in app.get("plotly_chart") if chart.proto.id.endswith(("removal_scores_baby", "removal_overlap_baby"))]

    def test_map_waits_for_open_click_and_inspector_selects_actual_fixture_item(self):
        app = self.app("atlas", liked=("H1", "H2"))
        self.assert_clean(app)
        self.assertEqual(self.calls["map"], [])
        self.assertEqual(len(app.get("plotly_chart")), 0)
        self.assertFalse(any(widget.label == "Inspect product" for widget in app.selectbox))
        self.button(app, "Open product atlas").click().run()
        self.assert_clean(app)
        self.assertEqual(len(self.calls["map"]), 1)
        self.assertEqual(self.calls["map"][0][:2], ("qwen_frozen", ("H1", "H2")))
        self.assertEqual(len(app.get("plotly_chart")), 1)
        self.assertEqual(self.selectbox(app, "Inspect product").value, "C1")
        self.selectbox(app, "Inspect product").set_value("B1").run()
        self.assert_clean(app)
        self.assertEqual(self.selectbox(app, "Inspect product").value, "B1")
        self.assertTrue(any(element.value == "DIAGNOSTIC_INSPECTOR: B1" for element in app.markdown))
        self.assertEqual(self.calls["removals"], [])

    def test_no_history_lab_has_empty_state_without_loading_relations_or_removals(self):
        app = self.app("lab")
        self.assert_clean(app)
        self.assertTrue(any("at least one liked product" in alert.value for alert in app.info))
        self.assertEqual(self.calls["relations"], [])
        self.assertEqual(self.calls["removals"], [])
        self.assertFalse(any(button.label == "Run removal experiment" for button in app.button))
        self.assertEqual(len(app.get("plotly_chart")), 0)

    def test_single_history_disables_removal_and_does_not_recompute_on_view_change(self):
        app = self.app("lab", liked=("H1",))
        self.assert_clean(app)
        self.assertTrue(any("Edges are not attention, modality weights or a causal decomposition" in caption.value for caption in app.caption))
        self.assertTrue(self.button(app, "Run removal experiment").disabled)
        self.assertTrue(any("At least two active history items" in alert.value for alert in app.info))
        self.assertEqual(self.calls["removals"], [])
        next(widget for widget in app.radio if widget.label == "Relationship view").set_value("Preference network").run()
        self.assert_clean(app)
        self.assertTrue(any("Edges are not attention, modality weights or a causal decomposition" in caption.value for caption in app.caption))
        self.assertTrue(self.button(app, "Run removal experiment").disabled)
        self.assertEqual(self.calls["removals"], [])
        self.assertEqual(len(app.get("plotly_chart")), 1)

    def test_two_histories_only_recompute_removal_after_explicit_run(self):
        app = self.app("lab", liked=("H1", "H2"))
        self.assert_clean(app)
        self.assertFalse(self.button(app, "Run removal experiment").disabled)
        self.assertEqual(self.calls["removals"], [])
        self.assertEqual(len(app.get("plotly_chart")), 1)
        self.selectbox(app, "Focus on a recommendation").set_value("C1").run()
        self.assert_clean(app)
        self.assertEqual(self.calls["removals"], [])
        self.button(app, "Run removal experiment").click().run()
        self.assert_clean(app)
        self.assertEqual(len(self.calls["removals"]), 1)
        self.assertEqual(self.calls["removals"][0][:2], ("qwen_frozen", ("H1", "H2")))
        self.assertEqual(len(app.get("plotly_chart")), 3)
        self.assertEqual(len(self.removal_charts(app)), 2)
        self.assertTrue(any("current session and seed42" in caption.value for caption in app.caption))

    def test_changing_model_invalidates_display_until_new_explicit_run(self):
        app = self.app("lab", liked=("H1", "H2"))
        self.assert_clean(app)
        self.button(app, "Run removal experiment").click().run()
        self.assert_clean(app)
        self.assertEqual(len(self.removal_charts(app)), 2)
        self.selectbox(app, "Experiment model").set_value("qwen_preference").run()
        self.assert_clean(app)
        self.assertEqual(len(self.calls["removals"]), 1)
        self.assertEqual(len(self.removal_charts(app)), 0)
        self.assertEqual(len(app.get("plotly_chart")), 1)
        self.assertTrue(any("Run again after changing preferences or the model" in caption.value for caption in app.caption))
        self.button(app, "Run removal experiment").click().run()
        self.assert_clean(app)
        self.assertEqual(len(self.calls["removals"]), 2)
        self.assertEqual(self.calls["removals"][-1][0], "qwen_preference")
        self.assertEqual(len(self.removal_charts(app)), 2)

    def test_changing_history_invalidates_display_until_new_explicit_run(self):
        app = self.app("lab", liked=("H1", "H2"))
        self.button(app, "Run removal experiment").click().run()
        self.assert_clean(app)
        self.assertEqual(len(self.removal_charts(app)), 2)
        app.session_state["diagnostic_likes"] = ["H2", "H3"]
        app.run()
        self.assert_clean(app)
        self.assertEqual(len(self.calls["removals"]), 1)
        self.assertEqual(len(self.removal_charts(app)), 0)
        self.assertTrue(any("Run again after changing preferences or the model" in caption.value for caption in app.caption))
        self.button(app, "Run removal experiment").click().run()
        self.assert_clean(app)
        self.assertEqual(len(self.calls["removals"]), 2)
        self.assertEqual(self.calls["removals"][-1][1], ("H2", "H3"))
        self.assertEqual(len(self.removal_charts(app)), 2)


if __name__ == "__main__":
    unittest.main()
