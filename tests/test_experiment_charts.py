"""The public evaluation is the fixture: charts must preserve its actual evidence."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from shoplens import experiment_charts as charts


SUMMARY = json.loads((Path(__file__).resolve().parents[1] / "submission" / "results_qwen_summary.json").read_text())


def training_rows(seed=42):
    rows = []
    evidence = SUMMARY["verification"]["unit_evidence"]
    for domain in charts.DOMAINS:
        for model in charts.MODELS:
            recorded = evidence[f"{domain}/{model}/seed{seed}"]
            curve = recorded["adaptation_curve"]
            rows.append({"domain": domain, "model": model, "seed": seed,
                         "epochs": [row["epoch"] for row in curve],
                         "validation_ndcg": [row["validation_ndcg@10"] for row in curve],
                         "selected_epoch": recorded["selected_epoch"]})
    return rows


def test_performance_preserves_all_nine_means_and_sample_sd_without_rescaling():
    recorded = {(row["domain"], row["model"]): row for row in SUMMARY["results"]}
    figure = charts.performance_figure(SUMMARY["results"])
    assert len(figure.data) == 9
    for trace, key in zip(figure.data, ((domain, model) for domain in charts.DOMAINS for model in charts.MODELS)):
        row = recorded[key]
        assert trace.x[0] == row["ndcg@10"]
        assert trace.error_x.array[0] == row["ndcg@10_std"]
        assert "sample SD" in trace.hovertemplate
        assert trace.customdata[0][1] == 3
    assert figure.layout.xaxis.range[0] == 0
    assert "no cross-domain averaging" in figure.layout.meta["interpretation"]


def test_metric_selector_uses_its_own_recorded_values_and_not_ndcg_defaults():
    rows = [row for row in SUMMARY["results"] if row["domain"] == "baby"]
    figure = charts.performance_figure(rows, "recall@20")
    frozen = next(row for row in rows if row["model"] == "qwen_frozen")
    assert len(figure.data) == 3
    assert figure.data[0].x[0] == frozen["recall@20"]
    assert figure.data[0].error_x.array[0] == frozen["recall@20_std"]
    assert "Recall@20" in figure.data[0].hovertemplate


@pytest.mark.parametrize("change", [
    lambda rows: rows.pop(),
    lambda rows: rows.append(copy.deepcopy(rows[0])),
    lambda rows: rows[0].update({"ndcg@10_std": -1}),
    lambda rows: rows[0].update({"ndcg@10": float("nan")}),
    lambda rows: rows[0].update({"seeds": [42, 43, 45]}),
])
def test_incomplete_invalid_or_unmatched_aggregates_are_not_plotted(change):
    rows = copy.deepcopy(SUMMARY["results"])
    change(rows)
    with pytest.raises(ValueError):
        charts.performance_figure(rows)


def test_heatmap_keeps_negative_signs_seed_identity_and_shared_zero_centre():
    figure = charts.paired_delta_figure(SUMMARY["paired_comparisons"])
    semantic_baby = next(row for row in SUMMARY["paired_comparisons"]
                         if row["domain"] == "baby" and row["before"] == "qwen_frozen" and row["after"] == "qwen_semantic")
    assert list(figure.data[0].z[0]) == semantic_baby["differences"]
    assert all(value < 0 for value in figure.data[0].z[0])
    assert figure.data[1].z[0][2] > 0  # Baby/Preference/seed44 is a recorded small win.
    assert figure.layout.coloraxis.cmin == -figure.layout.coloraxis.cmax
    assert figure.layout.coloraxis.cmid == 0
    assert list(figure.data[0].customdata[0][2]) == [44]


def test_forest_displays_real_seed_differences_and_paired_not_unpaired_sd():
    rows = [row for row in SUMMARY["paired_comparisons"] if row["domain"] == "baby"]
    figure = charts.paired_forest_figure(rows)
    row = next(row for row in rows if row["before"] == "qwen_frozen" and row["after"] == "qwen_semantic")
    assert list(figure.data[0].x) == row["differences"]
    assert figure.data[1].x[0] == row["mean_difference"]
    assert figure.data[1].error_x.array[0] == row["std_difference"]
    assert figure.layout.shapes[0].x0 == figure.layout.shapes[0].x1 == 0
    assert "not confidence intervals" in figure.layout.meta["interpretation"]


@pytest.mark.parametrize("change", [
    lambda rows: rows[0].update({"mean_difference": 0}),
    lambda rows: rows[0].update({"std_difference": 1}),
    lambda rows: rows[0].update({"seeds": [42, 42, 44]}),
    lambda rows: rows[0].update({"differences": [0]}),
    lambda rows: rows.pop(0),
])
def test_inconsistent_paired_summaries_are_rejected_instead_of_recomputed_silently(change):
    rows = copy.deepcopy(SUMMARY["paired_comparisons"])
    change(rows)
    with pytest.raises(ValueError):
        charts.paired_delta_figure(rows)


def test_no_paired_recall_or_nonexistent_metric_is_inferred():
    with pytest.raises(ValueError, match="selected metric"):
        charts.paired_delta_figure(SUMMARY["paired_comparisons"], "recall@10")
    with pytest.raises(ValueError, match="recorded NDCG or Recall"):
        charts.performance_figure(SUMMARY["results"], "accuracy")


@pytest.mark.parametrize("seed", [42, 43, 44])
def test_training_shows_only_recorded_adaptation_values_and_actual_selected_checkpoint(seed):
    rows = training_rows(seed)
    figure = charts.training_figure(rows)
    assert len(figure.data) == 18
    assert figure.layout.meta["seed"] == seed
    for index, row in enumerate(rows):
        curve, selected = figure.data[index * 2:index * 2 + 2]
        assert list(curve.x) == [1, 2]
        assert list(curve.y) == row["validation_ndcg"]
        assert selected.x[0] == row["selected_epoch"]
        assert selected.y[0] == row["validation_ndcg"][row["epochs"].index(row["selected_epoch"])]
    assert "no test points" in figure.layout.meta["interpretation"]


@pytest.mark.parametrize("change", [
    lambda rows: rows[0].update({"seed": 43}),
    lambda rows: rows[0].update({"selected_epoch": 3}),
    lambda rows: rows[0].update({"epochs": [2, 1]}),
    lambda rows: rows[0].update({"validation_ndcg": [.2]}),
    lambda rows: rows[0].update({"validation_ndcg": [.2, float("inf")]}),
])
def test_training_does_not_mix_seeds_or_invent_epochs(change):
    rows = training_rows()
    change(rows)
    with pytest.raises(ValueError):
        charts.training_figure(rows)


def test_one_domain_focus_keeps_all_three_models_and_never_adds_other_domains():
    rows = [row for row in training_rows() if row["domain"] == "sports"]
    figure = charts.training_figure(rows)
    assert len(figure.data) == 6
    assert [annotation.text for annotation in figure.layout.annotations] == ["Sports"]
