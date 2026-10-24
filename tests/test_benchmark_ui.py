"""Recorded benchmark routes must work without private runtime or encoders."""
from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


def app_for(view):
    return AppTest.from_string(f'''
from pathlib import Path
from unittest.mock import patch
import streamlit as st
from shoplens import ui
st.session_state["shoplens_workspace"] = "Benchmark lab"
st.session_state["benchmark_view"] = {view!r}
with patch.object(ui, "_root", return_value=Path("/nonexistent/shoplens-runtime")), \\
     patch.object(ui, "_items", side_effect=AssertionError("loaded private catalog")), \\
     patch.object(ui, "_available_models", side_effect=AssertionError("loaded trained runtime")), \\
     patch.object(ui, "_signature", side_effect=AssertionError("inspected runtime artifacts")):
    ui.main()
''', default_timeout=30).run()


@pytest.mark.parametrize("view", ["Search relevance", "Image + language", "Recommendation diagnostics", "Local performance"])
def test_all_portable_routes_read_real_records_without_runtime(view):
    app = app_for(view)
    assert not app.exception, [e.value for e in app.exception]
    assert not app.error, [e.value for e in app.error]
    assert len(app.get("plotly_chart")) >= 1
    assert len(app.metric) >= 4


def test_history_controls_keep_measured_domains_separate():
    app = app_for("Recommendation diagnostics")
    next(e for e in app.get("segmented_control") if e.label == "Diagnostic collection").set_value("baby").run()
    assert not app.exception
    frame = app.dataframe[0].value
    assert set(frame["Model"]) == {"Frozen", "Semantic", "Preference"}
    assert len(frame) == 3


def test_missing_results_show_missing_state_without_scores(tmp_path):
    app = AppTest.from_string(f'''
from pathlib import Path
from shoplens.benchmark_ui import render_benchmarks
render_benchmarks(Path({str(tmp_path)!r}))
''').run()
    assert not app.exception and not app.error
    assert len(app.info) == 1
    assert not app.metric and not app.get("plotly_chart")


def test_custom_modification_is_explicit_inference_without_target_ranks(monkeypatch):
    from shoplens import fashion_retrieval
    queries = __import__("json").loads((ROOT / "results/fashion/queries.json").read_text())
    query = next(q for q in queries if q["category"] == "dress")
    calls = []

    def encode(repo_root, runtime_root, reference_id, text, category):
        calls.append((reference_id, text, category))
        return {"candidate_count": 280, "labels_available": False,
                "methods": {m: {"top_ids": r["top_ids"][:3]} for m, r in query["methods"].items()}}

    monkeypatch.setattr(fashion_retrieval, "compose_fashion_query", encode)
    app = app_for("Image + language")
    next(e for e in app.radio if e.label == "Image + language interaction").set_value("Try my own modification").run()
    assert not calls
    assert not any(e.label == "Paired target rank" for e in app.metric)
    next(e for e in app.text_input if e.label == "Describe your modification").set_value("blue with longer sleeves")
    next(e for e in app.button if e.label == "Run my modification").click().run()
    assert not app.exception and not app.error
    assert calls == [(query["reference"], "blue with longer sleeves", "dress")]
    assert not any(e.label == "Paired target rank" for e in app.metric)
    # A different reference cannot inherit the preceding request's ranking.
    next(e for e in app.selectbox if e.label == "Recorded image + language request").set_value(1).run()
    assert len(calls) == 1
    assert not app.success
