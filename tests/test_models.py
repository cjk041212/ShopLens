"""Small synthetic fixtures validate behavior, never recommendation quality."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from shoplens.evaluate import evaluate_model, ranking_metrics
from shoplens.inference import available_models, explain, load_scorer, recommend
from shoplens.models import (HistoryRecommender, TemporalExamples, load_events,
                             pad_histories, stable_topk)
from shoplens.train import train_model


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


@pytest.fixture
def catalog(tmp_path):
    processed, artifacts = tmp_path / "processed", tmp_path / "artifacts"
    processed.mkdir()
    artifacts.mkdir()
    ids = [f"item{i:02d}" for i in range(30)]
    items = [{"item_id": item_id, "title": f"Product {index}",
              "text": f"product category{index % 4} material{index % 3} color{index % 5}"}
             for index, item_id in enumerate(ids)]
    events = []
    for user in range(8):
        for time, split in enumerate(("train", "train", "train", "validation", "test"), 1):
            events.append({"user_id": f"user{user}", "item_id": ids[(user * 3 + time) % len(ids)],
                           "timestamp": time * 100 + user, "rating": 5, "split": split})
    write_jsonl(processed / "items.jsonl", items)
    write_jsonl(processed / "events.jsonl", events)
    (processed / "audit.json").write_text(json.dumps({"source": "synthetic unit-test fixture", "synthetic": True}))
    rng = np.random.default_rng(12)
    np.savez_compressed(artifacts / "features.npz", item_ids=np.array(ids),
                        text=rng.normal(size=(30, 8)).astype(np.float32),
                        image=rng.normal(size=(30, 6)).astype(np.float32),
                        adapted_text=rng.normal(size=(30, 8)).astype(np.float32))
    (artifacts / "features.json").write_text(json.dumps({"synthetic": True, "purpose": "unit-test fixture"}))
    return processed, artifacts, ids


def test_strict_history_equal_time_repeat_and_limit():
    ids = [str(index) for index in range(7)]
    rows = [{"user_id": "u", "item_id": item, "timestamp": timestamp, "rating": rating, "split": split}
            for item, timestamp, rating, split in [
                ("0", 1, 5, "train"), ("1", 2, 5, "train"),
                ("2", 2, 5, "train"), ("3", 3, 1, "train"),
                ("4", 4, 5, "validation"), ("0", 5, 5, "test"),
                ("5", 6, 5, "test"), ("6", 7, 5, "test")]]
    protocol = TemporalExamples(rows, ids, max_history=2)
    train = protocol.examples("train")
    assert [row["history"] for row in train] == [[0], [0]]
    validation = protocol.examples("validation")[0]
    assert validation["target"] == 4
    assert validation["history"] == [2, 1]
    assert validation["seen"] == {0, 1, 2}
    test = protocol.examples("test")
    assert [row["target"] for row in test] == [5, 6]
    assert test[0]["history"] == [0, 4]
    assert test[1]["history"] == [5, 0]  # earlier test feedback enters later history
    assert 3 not in test[1]["seen"]  # low ratings do not become likes


def test_global_time_split_guard(tmp_path):
    rows = [{"user_id": "u", "item_id": "a", "rating": 5, "timestamp": 4, "split": "train"},
            {"user_id": "u", "item_id": "b", "rating": 5, "timestamp": 3, "split": "validation"}]
    write_jsonl(tmp_path / "events.jsonl", rows)
    with pytest.raises(ValueError, match="global chronological"):
        load_events(tmp_path)


def test_ranking_stable_ties_exclusions_and_nan():
    scores = np.array([0.5, 0.5, 0.5, 0.1, np.nan])
    assert stable_topk(scores, 2, {0}).tolist() == [1, 2]
    assert stable_topk(scores, 20, {0, 1, 2, 3}).tolist() == []
    assert stable_topk(scores, 0).tolist() == []


def test_macro_metrics_do_not_overweight_active_users():
    rows = [{"user_id": user, **{key: value for key in ("recall@10", "recall@20", "ndcg@10", "ndcg@20")}}
            for user, value in [("a", 1.0), ("a", 1.0), ("b", 0.0)]]
    metrics = ranking_metrics(rows)
    assert metrics["recall@10"] == 0.5
    assert metrics["n_targets"] == 3


def test_neural_forward_gradients_and_padding():
    rng = np.random.default_rng(1)
    model = HistoryRecommender(rng.normal(size=(6, 8)).astype(np.float32),
                               rng.normal(size=(6, 5)).astype(np.float32), embedding_dim=16)
    history = torch.from_numpy(pad_histories([[0, 1], [2]], 3))
    loss = model(history, torch.tensor([3, 4]), torch.tensor([4, 5]))
    loss.backward()
    assert torch.isfinite(loss)
    assert model.text_projection.weight.grad.abs().sum() > 0
    assert model.image_projection.weight.grad.abs().sum() > 0
    assert "text_features" not in model.state_dict()


@pytest.mark.parametrize("mode", ["popularity", "tfidf", "mf", "text", "multimodal", "adapted"])
def test_train_evaluate_and_new_user_inference(catalog, mode):
    processed, artifacts, ids = catalog
    metadata = train_model(processed, artifacts, mode, seed=7, epochs=2, batch_size=8, embedding_dim=16)
    assert mode in available_models(artifacts)
    recommendations = recommend(processed, artifacts, mode, ids[:3], k=5)
    assert len(recommendations) == 5
    assert not set(ids[:3]) & {row["item_id"] for row in recommendations}
    assert [row["rank"] for row in recommendations] == [1, 2, 3, 4, 5]
    details = explain(processed, artifacts, mode, ids[:3], recommendations[0]["item_id"])
    assert len(details["evidence"]) == 3
    result = evaluate_model(processed, artifacts, mode, seed=7, batch_size=3)
    assert result["n_targets"] == 8
    assert result["n_users"] == 8
    assert result["run_protocol"]["n_catalog_items"] == 30
    assert result["features_metadata"]["synthetic"] is True
    assert result["data_audit"]["synthetic"] is True
    assert 0 <= result["ndcg@10"] <= result["recall@10"] <= 1
    assert result["recall@20"] >= result["recall@10"]
    if mode not in {"tfidf", "popularity"}:
        assert metadata["selected_epoch"] == int(np.argmax(metadata["validation_ndcg@10"])) + 1


def test_test_labels_cannot_change_training_or_selection(catalog):
    processed, artifacts, ids = catalog
    original = train_model(processed, artifacts, "text", seed=11, epochs=2, batch_size=8, embedding_dim=16)
    original_state = torch.load(artifacts / "models" / "text.pt", weights_only=True)["state_dict"]
    events = load_events(processed)
    for row in events:
        if row["split"] == "test":
            row["item_id"] = ids[-1]
    write_jsonl(processed / "events.jsonl", events)
    altered = train_model(processed, artifacts, "text", seed=11, epochs=2, batch_size=8, embedding_dim=16)
    altered_state = torch.load(artifacts / "models" / "text.pt", weights_only=True)["state_dict"]
    assert original["validation_ndcg@10"] == altered["validation_ndcg@10"]
    assert all(torch.equal(original_state[key], altered_state[key]) for key in original_state)


def test_missing_artifact_and_feature_mismatch_fail_clearly(catalog):
    processed, artifacts, ids = catalog
    with pytest.raises(FileNotFoundError, match="train"):
        recommend(processed, artifacts, "text", ids[:3])
    train_model(processed, artifacts, "text", epochs=1, batch_size=8, embedding_dim=16)
    with np.load(artifacts / "features.npz") as source:
        values = dict(source)
    values["text"][0, 0] += 1
    np.savez_compressed(artifacts / "features.npz", **values)
    with pytest.raises(ValueError, match="feature values changed"):
        load_scorer(processed, artifacts, "text")


def test_adding_adapted_features_preserves_frozen_text_checkpoint(catalog):
    processed, artifacts, ids = catalog
    train_model(processed, artifacts, "text", epochs=1, batch_size=8, embedding_dim=16)
    before = recommend(processed, artifacts, "text", ids[:3])
    with np.load(artifacts / "features.npz") as source:
        values = dict(source)
    values["adapted_text"] += 0.5
    np.savez_compressed(artifacts / "features.npz", **values)
    assert recommend(processed, artifacts, "text", ids[:3]) == before


def test_low_interaction_partition_and_changed_split_guard(catalog):
    processed, artifacts, ids = catalog
    train_model(processed, artifacts, "popularity")
    result = evaluate_model(processed, artifacts, "popularity", low_interaction_threshold=1)
    assert result["low_interaction"]["n_targets"] + result["higher_interaction"]["n_targets"] == result["n_targets"]
    assert result["low_interaction"]["max_training_positive_users"] == 1
    events = load_events(processed)
    events[-1]["item_id"] = ids[-1]
    write_jsonl(processed / "events.jsonl", events)
    with pytest.raises(ValueError, match="different split"):
        evaluate_model(processed, artifacts, "popularity")


def test_all_likes_excluded_even_when_history_is_truncated(catalog):
    processed, artifacts, ids = catalog
    train_model(processed, artifacts, "popularity", max_history=2)
    rows = recommend(processed, artifacts, "popularity", ids[:5], k=30)
    assert len(rows) == 25
    assert not set(ids[:5]) & {row["item_id"] for row in rows}
