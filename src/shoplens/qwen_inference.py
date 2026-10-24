"""Serve exported Qwen experiment towers without loading the language model."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .models import digest, load_items, pad_histories, resolve_device, stable_topk

QWEN_MODES = ("qwen_frozen", "qwen_semantic", "qwen_preference")


def _paths(artifact_dir, model):
    directory = Path(artifact_dir)
    return (directory / "models" / f"{model}.json",
            directory / "models" / f"{model}.pt", directory / "serving.npz")


def available_models(artifact_dir):
    """Advertise only units whose training and formal evaluation both completed."""
    directory = Path(artifact_dir)
    if not (directory / "complete.json").is_file():
        return []
    return [mode for mode in QWEN_MODES if all(path.is_file() for path in _paths(directory, mode))]


class QwenScorer:
    """The same masked history average and user head used during training."""

    def __init__(self, processed_dir, artifact_dir, model, device="cpu"):
        if model not in QWEN_MODES:
            raise ValueError(f"Unknown Qwen model: {model}")
        self.processed_dir, self.artifact_dir = Path(processed_dir), Path(artifact_dir)
        self.mode, self.device = model, resolve_device(device)
        metadata_path, head_path, serving_path = _paths(artifact_dir, model)
        self.metadata = json.loads(metadata_path.read_text())
        if self.metadata["model"] != model or self.metadata.get("family") != "qwen_candidate_lora":
            raise ValueError("Qwen checkpoint model or family mismatch")
        self.items = load_items(self.processed_dir)
        self.item_ids = [str(item["item_id"]) for item in self.items]
        self.index = {item_id: index for index, item_id in enumerate(self.item_ids)}
        if self.metadata["item_ids"] != self.item_ids:
            raise ValueError("Saved Qwen catalog differs from current prepared data")
        if self.metadata["items_sha256"] != digest(self.processed_dir / "items.jsonl"):
            raise ValueError("Product metadata changed since Qwen training")
        for key, path in (("head_sha256", head_path), ("serving_sha256", serving_path)):
            if self.metadata.get(key) != digest(path):
                raise ValueError(f"Qwen artifact changed: {key}")
        self.max_history = int(self.metadata["max_history"])
        width = int(self.metadata["embedding_dim"])
        with np.load(serving_path, allow_pickle=False) as values:
            if list(values["item_ids"].astype(str)) != self.item_ids:
                raise ValueError("Serving catalog order differs from prepared data")
            history = np.asarray(values["history_item_vectors"], dtype=np.float32)
            candidate = np.asarray(values["candidate_item_vectors"], dtype=np.float32)
        for values in (history, candidate):
            if values.shape != (len(self.items), width) or not np.isfinite(values).all():
                raise ValueError("Invalid Qwen serving vectors")
        self.history_item_vectors = torch.as_tensor(history, device=self.device)
        self.item_vectors = torch.as_tensor(candidate, device=self.device)
        self.user_network = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, width))
        state = torch.load(head_path, map_location="cpu", weights_only=True)["state_dict"]
        user_state = {key.removeprefix("user_network."): value
                      for key, value in state.items() if key.startswith("user_network.")}
        self.user_network.load_state_dict(user_state, strict=True)
        self.user_network.to(self.device).eval()

    def score_histories(self, histories):
        if not histories or any(not history for history in histories):
            raise ValueError("At least one liked item is needed for every history")
        if any(index < 0 or index >= len(self.items) for history in histories for index in history):
            raise ValueError("History contains an unknown catalog index")
        padded = torch.as_tensor(pad_histories(histories, self.max_history), device=self.device)
        mask = padded >= 0
        with torch.inference_mode():
            vectors = self.history_item_vectors[padded.clamp_min(0)]
            mean = (vectors * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
            users = F.normalize(self.user_network(mean), dim=-1)
            return (users @ self.item_vectors.T).float().cpu().numpy()


@lru_cache(maxsize=9)
def _cached_scorer(processed_dir, artifact_dir, model, signature):
    return QwenScorer(processed_dir, artifact_dir, model)


def load_scorer(processed_dir, artifact_dir, model):
    processed_dir, artifact_dir = Path(processed_dir).resolve(), Path(artifact_dir).resolve()
    paths = (processed_dir / "items.jsonl", *_paths(artifact_dir, model))
    if any(not path.is_file() for path in paths):
        raise FileNotFoundError(f"Qwen unit {model} has no complete serving export in {artifact_dir}")
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
    liked = _liked_indices(scorer, liked_item_ids)
    scores = scorer.score_histories([liked])[0]
    ranking = stable_topk(scores, k, liked)
    return [{"item_id": scorer.item_ids[index], "rank": rank, "score": float(scores[index])}
            for rank, index in enumerate(ranking, 1)]


def explain(processed_dir, artifact_dir, model, liked_item_ids, item_id):
    scorer = load_scorer(processed_dir, artifact_dir, model)
    liked = _liked_indices(scorer, liked_item_ids)[:scorer.max_history]
    if item_id not in scorer.index:
        raise ValueError(f"Unknown item: {item_id}")
    target = scorer.index[item_id]
    with torch.inference_mode():
        similarities = (scorer.history_item_vectors[liked] @ scorer.item_vectors[target]).cpu().numpy()
    evidence = [{"item_id": scorer.item_ids[index], "title": scorer.items[index].get("title", ""),
                 "text": str(scorer.items[index].get("text", ""))[:500], "similarity": float(similarity)}
                for index, similarity in zip(liked, similarities)]
    evidence.sort(key=lambda row: -row["similarity"])
    return {"model": model, "item_id": item_id,
            "score": float(scorer.score_histories([liked])[0, target]), "evidence": evidence,
            "training_positive_users": int(scorer.metadata["training_item_counts"][target]),
            "note": "History-to-candidate tower similarities show input evidence; they are not a causal explanation or a decomposition of the user-network score."}
