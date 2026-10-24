"""Result-shaped synthetic fixtures exercise analysis; they are not measurements."""

import copy
import json

import pytest

from shoplens.report_analysis import build_analysis


def result(mode, seed, value):
    return {
        "model": mode, "split": "test", "seed": seed,
        "ndcg@10": value, "recall@10": value + .1,
        "ndcg@20": value + .05, "recall@20": value + .2,
        "n_users": 12, "n_targets": 20, "selected_epoch": 4,
        "items_sha256": "fixture-items", "events_sha256": "fixture-events",
        "selected_features_sha256": None if mode == "mf" else "fixture-features-" + mode,
        "data_audit": {"synthetic": False, "scope": "full-input", "subset": {}},
        "features_metadata": {"synthetic": False},
        "training_configuration": {"epochs": 5, "max_examples": None},
        "run_protocol": {
            "candidate_set": "full catalog minus past likes", "history": "strictly earlier likes",
            "max_history": 20, "evaluation": "prequential", "averaging": "user macro",
            "positive_feedback": "rating >=4", "repeat_targets": "skip", "ties": "item-id",
            "n_catalog_items": 100, "evaluation_subset": False,
            "mf_inference": "ridge history fold-in" if mode == "mf" else None,
        },
        "low_interaction": {
            "max_training_positive_users": 5, "n_users": 4, "n_targets": 6,
            "ndcg@10": value / 2, "recall@10": value / 2 + .05,
            "ndcg@20": value / 2 + .02, "recall@20": value / 2 + .1,
        },
    }


def write_results(root, rows):
    directory = root / "artifacts" / "baby"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "results.json").write_text(json.dumps({"results": rows}))


def test_seed_matched_delta_and_low_group_table(tmp_path):
    rows = [result(mode, seed, value)
            for mode, values in [("text", [.1, .2, .3]), ("multimodal", [.2, .1, .4]),
                                 ("adapted", [.19, .12, .39])]
            for seed, value in zip((42, 43, 44), values)]
    write_results(tmp_path, rows)
    report = build_analysis(tmp_path)
    assert "+0.033333 ± 0.115470" in report
    assert "2／0／1" in report
    assert "42, 43, 44" in report
    assert "不足以给出统计显著性结论" in report
    assert "低交互商品检查" in report and "4／6" in report
    assert "MF 训练预算敏感性检查" not in report


def test_only_intersection_of_seeds_is_paired(tmp_path):
    rows = [result("text", 42, .1), result("text", 43, .8), result("multimodal", 42, .2)]
    write_results(tmp_path, rows)
    report = build_analysis(tmp_path)
    assert "+0.100000（单次，无标准差）" in report
    assert "未匹配种子" in report
    assert "+0.350000" not in report


@pytest.mark.parametrize("change", [
    lambda row: row["data_audit"].update(synthetic=True),
    lambda row: row["features_metadata"].update(synthetic=True),
    lambda row: row["run_protocol"].update(evaluation_subset=True),
    lambda row: row.update(ndcg__unused=0, **{"ndcg@10": float("nan")}),
    lambda row: row["training_configuration"].update(max_examples=5),
])
def test_synthetic_diagnostic_and_nonfinite_results_rejected(tmp_path, change):
    row = result("text", 42, .1)
    change(row)
    write_results(tmp_path, [row])
    with pytest.raises(ValueError):
        build_analysis(tmp_path)


@pytest.mark.parametrize("change", [
    lambda row: row.update(events_sha256="different-events"),
    lambda row: row["run_protocol"].update(candidate_set="sampled negatives"),
    lambda row: row["data_audit"].update(scope="subset", subset={"max_users": 100}),
    lambda row: row["low_interaction"].update(max_training_positive_users=10),
    lambda row: row["low_interaction"].update(n_targets=5),
])
def test_cross_method_pooling_requires_identical_protocol(tmp_path, change):
    first, second = result("text", 42, .1), result("multimodal", 42, .2)
    change(second)
    write_results(tmp_path, [first, second])
    with pytest.raises(ValueError):
        build_analysis(tmp_path)


def test_duplicate_seed_rejected(tmp_path):
    row = result("text", 42, .1)
    write_results(tmp_path, [row, copy.deepcopy(row)])
    with pytest.raises(ValueError, match="Duplicate seed"):
        build_analysis(tmp_path)


def write_budget(root, extended):
    directory = root / "artifacts" / "budget_check" / "baby" / "seed42"
    directory.mkdir(parents=True)
    (directory / "budget_check.json").write_text(json.dumps({
        "domain": "baby", "seed": 42, "epochs": 20, "selected_epoch": 15,
        "validation_curve": [.1] * 20, "result": extended,
    }))


def test_extra_mf_budget_is_separate_and_matched(tmp_path):
    original = result("mf", 42, .1)
    extended = result("mf", 42, .25)
    extended["training_configuration"]["epochs"] = 20
    extended["selected_epoch"] = 15
    write_results(tmp_path, [original])
    write_budget(tmp_path, extended)
    report = build_analysis(tmp_path)
    assert "MF 训练预算敏感性检查" in report
    assert "5／20" in report
    assert "+0.150000" in report
    assert "不替换主表" in report


def test_budget_check_cannot_mix_candidate_protocol(tmp_path):
    original, extended = result("mf", 42, .1), result("mf", 42, .25)
    extended["training_configuration"]["epochs"] = 20
    extended["selected_epoch"] = 15
    extended["run_protocol"]["candidate_set"] = "sampled negatives"
    write_results(tmp_path, [original])
    write_budget(tmp_path, extended)
    with pytest.raises(ValueError, match="candidate/history"):
        build_analysis(tmp_path)


def test_empty_low_cohort_is_not_fabricated_as_zero(tmp_path):
    row = result("text", 42, .1)
    row["low_interaction"].update(n_users=0, n_targets=0)
    for key in ("ndcg@10", "recall@10", "ndcg@20", "recall@20"):
        row["low_interaction"][key] = None
    write_results(tmp_path, [row])
    assert "无有效目标" in build_analysis(tmp_path)
