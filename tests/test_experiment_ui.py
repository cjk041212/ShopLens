"""Real published summary drives UI tests; model/runtime files are never needed."""

from __future__ import annotations

import copy
import json
import tempfile
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest


SOURCE_ROOT = Path(__file__).resolve().parents[1]
SUMMARY = json.loads((SOURCE_ROOT / "results/qwen/results_qwen_summary.json").read_text())


def control(app, group, label):
    return next(element for element in getattr(app, group) if element.label == label)


def clean(app):
    assert not app.exception, [element.value for element in app.exception]
    assert not app.error, [element.value for element in app.error]


def gallery_app(empty_runtime):
    # Calling the actual main route protects the no-weights early return. These
    # boundaries fail if the gallery accidentally starts product/model loading.
    return AppTest.from_string(f'''
from pathlib import Path
from unittest.mock import patch
import streamlit as st
from shoplens import ui
st.session_state.setdefault("shoplens_workspace", "Experiment gallery")
with patch.object(ui, "_root", return_value=Path({str(empty_runtime)!r})), \\
     patch.object(ui, "_items", side_effect=AssertionError("Gallery loaded product vectors")), \\
     patch.object(ui, "_available_models", side_effect=AssertionError("Gallery loaded models")), \\
     patch.object(ui, "_signature", side_effect=AssertionError("Gallery inspected runtime artifacts")):
    ui.main()
''', default_timeout=30).run()


def test_actual_gallery_route_loads_real_public_scores_with_empty_runtime():
    with tempfile.TemporaryDirectory() as directory:
        app = gallery_app(Path(directory))
        clean(app)
        assert list(Path(directory).iterdir()) == []
        assert len(app.get("plotly_chart")) == 1
        values = app.dataframe[0].value
        assert len(values) == 9
        assert set(values["domain"]) == {"baby", "sports", "clothing"}
        for row in values.to_dict("records"):
            actual = next(source for source in SUMMARY["results"]
                          if (source["domain"], source["model"]) == (row["domain"], row["model"]))
            assert row["mean"] == actual["ndcg@10"]
            assert row["sample_sd"] == actual["ndcg@10_std"]
            assert row["n_seeds"] == 3
        assert any("standard deviations are not confidence intervals" in element.value for element in app.markdown)


def test_collection_and_metric_filter_keep_exact_domain_records_separate():
    with tempfile.TemporaryDirectory() as directory:
        app = gallery_app(Path(directory))
        control(app, "selectbox", "Collection scope").set_value("clothing").run()
        control(app, "selectbox", "Test metric").set_value("recall@20").run()
        clean(app)
        values = app.dataframe[0].value
        assert len(values) == 3
        assert set(values["domain"]) == {"clothing"}
        assert set(values["metric"]) == {"recall@20"}
        for row in values.to_dict("records"):
            source = next(record for record in SUMMARY["results"]
                          if record["domain"] == "clothing" and record["model"] == row["model"])
            assert row["mean"] == source["recall@20"]
            assert row["sample_sd"] == source["recall@20_std"]


def test_paired_views_stay_on_recorded_ndcg10_despite_other_metric_selector():
    with tempfile.TemporaryDirectory() as directory:
        app = gallery_app(Path(directory))
        control(app, "selectbox", "Collection scope").set_value("baby").run()
        control(app, "selectbox", "Test metric").set_value("recall@20").run()
        control(app, "radio", "Explore the experiment").set_value("Paired effects").run()
        for style in ("Seed matrix", "Effect overview"):
            control(app, "radio", "Paired view").set_value(style).run()
            clean(app)
            values = app.dataframe[0].value
            assert len(values) == 6
            assert set(values["metric"]) == {"ndcg@10"}
            assert set(values["domain"]) == {"baby"}
            assert set(values["reference"]) == {"qwen_frozen"}
            for row in values.to_dict("records"):
                source = next(record for record in SUMMARY["paired_comparisons"]
                              if record["domain"] == "baby" and record["before"] == "qwen_frozen"
                              and record["after"] == row["model"])
                assert row["difference"] == source["differences"][source["seeds"].index(row["seed"])]
            assert control(app, "selectbox", "Test metric").disabled
            assert control(app, "selectbox", "Test metric").value == "ndcg@10"


def test_training_seed_and_checkpoint_export_are_actual_recorded_validation():
    with tempfile.TemporaryDirectory() as directory:
        app = gallery_app(Path(directory))
        control(app, "selectbox", "Collection scope").set_value("sports").run()
        control(app, "radio", "Explore the experiment").set_value("Training dynamics").run()
        for seed in (42, 43, 44):
            control(app, "radio", "Training seed").set_value(seed).run()
            clean(app)
            values = app.dataframe[0].value
            assert len(values) == 6
            assert set(values["domain"]) == {"sports"}
            assert set(values["seed"]) == {seed}
            assert "test_ndcg@10" not in values.columns
            for row in values.to_dict("records"):
                source = SUMMARY["verification"]["unit_evidence"][f"sports/{row['model']}/seed{seed}"]
                expected = source["adaptation_curve"][row["epoch"] - 1]["validation_ndcg@10"]
                assert row["validation_ndcg@10"] == expected
                assert row["selected_checkpoint"] == (row["epoch"] == source["selected_epoch"])


@pytest.mark.parametrize("failure", ["missing", "incomplete", "bad_number"])
def test_invalid_summary_shows_error_and_does_not_create_scores(failure):
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        if failure != "missing":
            document = copy.deepcopy(SUMMARY)
            if failure == "incomplete":
                document["verification"]["complete"] = False
            else:
                document["results"][0]["ndcg@10"] = "not a recorded number"
            (root / "results/qwen").mkdir(parents=True)
            (root / "results/qwen/results_qwen_summary.json").write_text(json.dumps(document))
        app = AppTest.from_string(f'''
from shoplens.experiment_ui import render_gallery
render_gallery({str(root)!r})
''', default_timeout=30).run()
        assert not app.exception
        assert len(app.error) == 1
        assert "summary is unavailable" in app.error[0].value
        assert not app.dataframe
        assert not app.get("plotly_chart")
        assert not app.get("download_button")
