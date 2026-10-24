"""Load fitted artifacts and recommend to a session containing liked products."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.preprocessing import normalize

from .models import (MODES, NEURAL_MODES, HistoryRecommender, MatrixFactorization,
                     digest, feature_content_digest, load_features, load_items, pad_histories, resolve_device, stable_topk)


def artifact_path(artifact_dir, model):
    extension = ".npz" if model == "popularity" else ".joblib" if model == "tfidf" else ".pt"
    return Path(artifact_dir) / "models" / f"{model}{extension}"


def available_models(artifact_dir):
    directory = Path(artifact_dir) / "models"
    return [mode for mode in MODES if (directory / f"{mode}.json").is_file() and artifact_path(artifact_dir, mode).is_file()]


class FittedScorer:
    def __init__(self, processed_dir, artifact_dir, model, device="cpu"):
        self.processed_dir, self.artifact_dir = Path(processed_dir), Path(artifact_dir)
        self.mode = model
        self.device = resolve_device(device)
        if model not in MODES:
            raise ValueError(f"Unknown model: {model}")
        if model not in available_models(artifact_dir):
            raise FileNotFoundError(f"Model {model!r} has not been trained in {artifact_dir}; run python -m shoplens.train first")
        self.items = load_items(processed_dir)
        self.item_ids = [str(row["item_id"]) for row in self.items]
        self.index = {item_id: index for index, item_id in enumerate(self.item_ids)}
        self.metadata = json.loads((self.artifact_dir / "models" / f"{model}.json").read_text())
        if self.metadata["item_ids"] != self.item_ids:
            raise ValueError("Saved model catalog differs from the current processed data; retrain")
        if self.metadata["items_sha256"] != digest(self.processed_dir / "items.jsonl"):
            raise ValueError("Product metadata changed since training; retrain the model")
        self.max_history = int(self.metadata.get("max_history", 20))
        path = artifact_path(artifact_dir, model)
        if model == "popularity":
            with np.load(path, allow_pickle=False) as values:
                self.counts = values["counts"].astype(np.float32)
        elif model == "tfidf":
            fitted = joblib.load(path)
            self.matrix = fitted["matrix"]
        else:
            state = torch.load(path, map_location="cpu", weights_only=True)["state_dict"]
            if model in NEURAL_MODES:
                text, image = load_features(artifact_dir, self.item_ids, model)
                if self.metadata.get("selected_features_sha256") != feature_content_digest(text, image):
                    raise ValueError("Selected feature values changed since training; retrain with the current features")
                self.network = HistoryRecommender(text, image, self.metadata["embedding_dim"], self.max_history)
                self.network.load_state_dict(state)
                self.network.to(self.device)
                self.network.eval()
                with torch.inference_mode():
                    self.item_vectors = self.network.item_vectors()
            else:
                network = MatrixFactorization(len(self.metadata["user_ids"]), len(self.items), self.metadata["embedding_dim"])
                network.load_state_dict(state)
                self.item_vectors = network.items.weight.detach().numpy()
                # Precompute the small normal-equation inverse once per load.
                values = self.item_vectors.astype(np.float64)
                self.inverse = np.linalg.inv(values.T @ values + np.eye(values.shape[1]))

    def score_histories(self, histories):
        """Return a [batch, full catalog] matrix; no sampled evaluation negatives."""
        if not histories or any(not history for history in histories):
            raise ValueError("At least one liked item is needed for every history")
        histories = [list(history)[:self.max_history] for history in histories]
        if self.mode == "popularity":
            return np.broadcast_to(self.counts, (len(histories), len(self.items))).copy()
        if self.mode == "tfidf":
            from scipy.sparse import csr_matrix, vstack
            users = vstack([csr_matrix(self.matrix[history].mean(axis=0)) for history in histories]).tocsr()
            return (normalize(users) @ self.matrix.T).toarray()
        if self.mode in NEURAL_MODES:
            with torch.inference_mode():
                users = self.network.user_vectors(torch.from_numpy(pad_histories(histories, self.max_history)).to(self.device), self.item_vectors)
                return (users @ self.item_vectors.T).cpu().numpy()
        sums = np.stack([self.item_vectors[history].sum(axis=0) for history in histories])
        return (sums @ self.inverse @ self.item_vectors.T).astype(np.float32)


@lru_cache(maxsize=12)
def _cached_scorer(processed_dir, artifact_dir, model, signature):
    return FittedScorer(processed_dir, artifact_dir, model)


def load_scorer(processed_dir, artifact_dir, model):
    processed_dir, artifact_dir = Path(processed_dir).resolve(), Path(artifact_dir).resolve()
    paths = [processed_dir / "items.jsonl", artifact_dir / "models" / f"{model}.json", artifact_path(artifact_dir, model)]
    if model in NEURAL_MODES:
        paths.append(artifact_dir / "features.npz")
    if any(not path.exists() for path in paths):
        raise FileNotFoundError(f"Missing data/model artifact for {model!r}; prepare data and train it first")
    signature = tuple((path.stat().st_mtime_ns, path.stat().st_size) for path in paths)
    return _cached_scorer(str(processed_dir), str(artifact_dir), model, signature)


def _liked_indices(scorer, liked_item_ids):
    unknown = set(liked_item_ids) - set(scorer.index)
    if unknown:
        raise ValueError(f"Unknown liked item IDs: {sorted(unknown)[:5]}")
    ids = list(dict.fromkeys(liked_item_ids))
    if not ids:
        raise ValueError("Select at least one liked product first")
    return [scorer.index[item_id] for item_id in reversed(ids)]


def recommend(processed_dir, artifact_dir, model, liked_item_ids, k=10):
    scorer = load_scorer(processed_dir, artifact_dir, model)
    indices = _liked_indices(scorer, liked_item_ids)
    scores = scorer.score_histories([indices])[0]
    ranking = stable_topk(scores, k, indices)
    return [{"item_id": scorer.item_ids[index], "score": float(scores[index]), "rank": rank}
            for rank, index in enumerate(ranking, 1)]


def explain(processed_dir, artifact_dir, model, liked_item_ids, item_id):
    scorer = load_scorer(processed_dir, artifact_dir, model)
    liked = _liked_indices(scorer, liked_item_ids)[:scorer.max_history]
    if item_id not in scorer.index:
        raise ValueError(f"Unknown item: {item_id}")
    target = scorer.index[item_id]
    score = float(scorer.score_histories([liked])[0, target])
    evidence = [{"item_id": scorer.item_ids[index], "title": scorer.items[index].get("title", ""),
                 "text": str(scorer.items[index].get("text", ""))[:500]} for index in liked]
    if model == "tfidf":
        similarities = (scorer.matrix[liked] @ scorer.matrix[target].T).toarray().ravel()
        for row, similarity in zip(evidence, similarities):
            row["text_similarity"] = float(similarity)
            row["similarity"] = float(similarity)
    elif model in NEURAL_MODES:
        text = scorer.network.text_features.numpy()
        image = None if scorer.network.image_features is None else scorer.network.image_features.numpy()
        for name, values in [("text_similarity", text), ("image_similarity", image)]:
            if values is None:
                continue
            unit = values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)
            similarities = unit[liked] @ unit[target]
            for row, similarity in zip(evidence, similarities):
                row[name] = float(similarity)
        with torch.inference_mode():
            similarities = (scorer.item_vectors[liked] @ scorer.item_vectors[target]).numpy()
        for row, similarity in zip(evidence, similarities):
            row["similarity"] = float(similarity)
    elif model == "mf":
        values = scorer.item_vectors
        unit = values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)
        for row, similarity in zip(evidence, unit[liked] @ unit[target]):
            row["similarity"] = float(similarity)
    evidence.sort(key=lambda row: -row.get("similarity", 0.0))
    return {"model": model, "item_id": item_id, "score": score, "evidence": evidence,
            "note": "Evidence shows input-history items and feature similarities; it is not a causal explanation of the neural score.",
            "training_positive_users": int(scorer.metadata["training_item_counts"][target])}
