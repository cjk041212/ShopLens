"""Asymmetric ranker: immutable base-history inputs, adaptable candidate inputs."""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .models import pad_histories, stable_topk


class QwenHistoryRecommender(nn.Module):
    def __init__(self, base_text, image, candidate_text=None, embedding_dim=128, max_history=20):
        super().__init__()
        self.embedding_dim, self.max_history = embedding_dim, max_history
        self.register_buffer("base_text", self._normalized(base_text), persistent=False)
        self.register_buffer("images", self._normalized(image), persistent=False)
        self.register_buffer("candidate_text", self._normalized(base_text if candidate_text is None else candidate_text), persistent=False)
        text_width, image_width = base_text.shape[1], image.shape[1]
        self.history_text_projection = nn.Linear(text_width, embedding_dim)
        self.history_image_projection = nn.Linear(image_width, embedding_dim)
        self.history_fusion = nn.Sequential(nn.Linear(embedding_dim * 2, embedding_dim), nn.GELU(), nn.LayerNorm(embedding_dim))
        self.candidate_text_projection = nn.Linear(text_width, embedding_dim)
        self.candidate_image_projection = nn.Linear(image_width, embedding_dim)
        self.candidate_fusion = nn.Sequential(nn.Linear(embedding_dim * 2, embedding_dim), nn.GELU(), nn.LayerNorm(embedding_dim))
        self.user_network = nn.Sequential(nn.Linear(embedding_dim, embedding_dim), nn.GELU(), nn.Linear(embedding_dim, embedding_dim))

    @staticmethod
    def _normalized(values):
        return F.normalize(torch.as_tensor(values, dtype=torch.float32), dim=-1)

    def set_candidate_text(self, values):
        tensor = torch.as_tensor(values, dtype=torch.float32, device=self.base_text.device)
        if tensor.shape != self.base_text.shape or not torch.isfinite(tensor).all():
            raise ValueError("Candidate feature shape or values differ from the base catalog")
        self.candidate_text = F.normalize(tensor.detach(), dim=-1)

    def item_vectors(self, indices=None, text_override=None):
        text = self.candidate_text if indices is None else self.candidate_text[indices]
        if text_override is not None:
            text = F.normalize(text_override.float(), dim=-1)
        image = self.images if indices is None else self.images[indices]
        joined = torch.cat((self.candidate_text_projection(text), self.candidate_image_projection(image)), dim=-1)
        return F.normalize(self.candidate_fusion(joined), dim=-1)

    def history_item_vectors(self, indices=None):
        text = self.base_text if indices is None else self.base_text[indices]
        image = self.images if indices is None else self.images[indices]
        joined = torch.cat((self.history_text_projection(text), self.history_image_projection(image)), dim=-1)
        return F.normalize(self.history_fusion(joined), dim=-1)

    def user_vectors(self, histories, history_vectors=None):
        mask = histories >= 0
        safe = histories.clamp_min(0)
        values = self.history_item_vectors(safe) if history_vectors is None else history_vectors[safe]
        pooled = (values * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
        return F.normalize(self.user_network(pooled), dim=-1)

    def forward(self, histories, positive_indices, negative_indices, positive_text=None, negative_text=None):
        user = self.user_vectors(histories)
        positive = self.item_vectors(positive_indices, positive_text)
        negative = self.item_vectors(negative_indices, negative_text)
        difference = (user * (positive - negative)).sum(-1)
        return -F.logsigmoid(5.0 * difference).mean()


def multi_positive_infonce(anchors, documents, item_indices, temperature=.07):
    """All same-ASIN pairs are positives in both directions, including duplicates."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    anchors, documents = F.normalize(anchors.float(), dim=-1), F.normalize(documents.float(), dim=-1)
    labels = torch.as_tensor(item_indices, device=documents.device)
    positives = labels[:, None] == labels[None, :]
    logits = anchors @ documents.T / temperature
    positive_logits = logits.masked_fill(~positives, -torch.inf)
    row_loss = torch.logsumexp(logits, 1) - torch.logsumexp(positive_logits, 1)
    column_loss = torch.logsumexp(logits, 0) - torch.logsumexp(positive_logits, 0)
    return (row_loss.mean() + column_loss.mean()) / 2


def validate_full_catalog(model, examples, batch_size=128):
    """User-macro NDCG@10; current candidate cache must match this checkpoint."""
    if not examples:
        raise ValueError("No eligible downstream validation cases")
    was_training = model.training
    model.eval()
    users = defaultdict(list)
    with torch.inference_mode():
        candidates, history = model.item_vectors(), model.history_item_vectors()
        for start in range(0, len(examples), batch_size):
            batch = examples[start:start + batch_size]
            padded = torch.from_numpy(pad_histories([row["history"] for row in batch], model.max_history)).to(candidates.device)
            scores = (model.user_vectors(padded, history) @ candidates.T).cpu().numpy()
            for row, values in zip(batch, scores):
                rank = stable_topk(values, 10, row["seen"])
                match = np.flatnonzero(rank == row["target"])
                users[row["user_id"]].append(0.0 if not len(match) else float(1 / np.log2(int(match[0]) + 2)))
    model.train(was_training)
    return float(np.mean([np.mean(values) for values in users.values()]))


def export_serving(model):
    was_training = model.training
    model.eval()
    with torch.inference_mode():
        result = {"history_item_vectors": model.history_item_vectors().cpu().numpy().astype(np.float32),
                  "candidate_item_vectors": model.item_vectors().cpu().numpy().astype(np.float32)}
    model.train(was_training)
    return result
