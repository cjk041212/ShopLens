"""Small, inspectable recommendation models and the shared temporal protocol."""

from __future__ import annotations

import hashlib
import itertools
import json
import random
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

MODES = ("popularity", "mf", "tfidf", "text", "multimodal", "adapted")
NEURAL_MODES = ("text", "multimodal", "adapted")


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def load_items(processed_dir):
    items = read_jsonl(Path(processed_dir) / "items.jsonl")
    ids = [str(row["item_id"]) for row in items]
    if ids != sorted(ids) or len(ids) != len(set(ids)):
        raise ValueError("items.jsonl must have unique item_id values in sorted order")
    return items


def load_events(processed_dir):
    events = read_jsonl(Path(processed_dir) / "events.jsonl")
    for event in events:
        if event.get("split") not in {"train", "validation", "test"}:
            raise ValueError("Every event needs its prepared train/validation/test split")
    ranges = {split: [int(row["timestamp"]) for row in events if row["split"] == split]
              for split in ("train", "validation", "test")}
    nonempty = [values for values in ranges.values() if values]
    if any(max(left) > min(right) for left, right in zip(nonempty, nonempty[1:])):
        raise ValueError("Prepared splits must follow one global chronological cutoff order")
    return sorted(events, key=lambda row: (int(row["timestamp"]), str(row["user_id"]), str(row["item_id"])))


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def feature_content_digest(text, image=None):
    value = hashlib.sha256()
    for array in (text, image):
        if array is None:
            value.update(b"none")
        else:
            array = np.ascontiguousarray(array, dtype=np.float32)
            value.update(str(array.shape).encode())
            value.update(array.tobytes())
    return value.hexdigest()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def resolve_device(device):
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; choose --device cpu or a CUDA host")
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable; choose --device cpu")
    return torch.device(device)


def load_features(artifact_dir, item_ids, mode):
    path = Path(artifact_dir) / "features.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run feature encoding first")
    with np.load(path, allow_pickle=False) as data:
        if list(data["item_ids"].astype(str)) != list(item_ids):
            raise ValueError("Feature item_ids do not match the sorted processed catalog")
        text_key = "adapted_text" if mode == "adapted" else "text"
        if text_key not in data:
            raise ValueError(f"Features lack {text_key}; encode the adapted model first")
        text = np.asarray(data[text_key], dtype=np.float32)
        image = np.asarray(data["image"], dtype=np.float32) if mode != "text" else None
    if text.ndim != 2 or len(text) != len(item_ids) or not np.isfinite(text).all():
        raise ValueError("Invalid text feature matrix")
    if image is not None and (image.ndim != 2 or len(image) != len(item_ids) or not np.isfinite(image).all()):
        raise ValueError("Invalid image feature matrix")
    return text, image


class TemporalExamples:
    """Targets use only strictly earlier positive events, never equal-time events.

    Evaluation is prequential: weights remain fixed, but an earlier validation/test
    interaction may become history for a later target. Repeat purchases are skipped
    because the protocol ranks unseen items. Histories retain at most max_history
    distinct, most recently liked items; exclusions include the complete past.
    """

    def __init__(self, events, item_ids, max_history=20):
        self.item_index = {item_id: i for i, item_id in enumerate(item_ids)}
        self.max_history = max_history
        by_user = defaultdict(list)
        for row in events:
            if float(row["rating"]) >= 4 and str(row["item_id"]) in self.item_index:
                by_user[str(row["user_id"])].append(row)
        self.by_user = {}
        for user, rows in by_user.items():
            rows.sort(key=lambda row: (int(row["timestamp"]), str(row["item_id"])))
            self.by_user[user] = rows

    def examples(self, split, include_seen=True):
        examples = []
        for user in sorted(self.by_user):
            rows = self.by_user[user]
            seen, recent = set(), OrderedDict()
            for timestamp, grouped in itertools.groupby(rows, key=lambda row: int(row["timestamp"])):
                group = list(grouped)
                for row in group:
                    target = self.item_index[str(row["item_id"])]
                    if row["split"] == split and seen and target not in seen:
                        examples.append({"user_id": user, "target": target,
                                         "history": list(reversed(recent)),
                                         "seen": seen.copy() if include_seen else None,
                                         "timestamp": timestamp})
                # Equal-time targets are all evaluated before any can enter history.
                for row in group:
                    index = self.item_index[str(row["item_id"])]
                    seen.add(index)
                    recent.pop(index, None)
                    recent[index] = None
                    if len(recent) > self.max_history:
                        recent.popitem(last=False)
        return examples


def pad_histories(histories, max_history=20):
    result = np.full((len(histories), max_history), -1, dtype=np.int64)
    for index, history in enumerate(histories):
        values = list(history)[:max_history]
        result[index, :len(values)] = values
    return result


def training_counts(events, item_ids):
    index = {item_id: i for i, item_id in enumerate(item_ids)}
    counts = np.zeros(len(item_ids), dtype=np.int64)
    unique = set()
    for row in events:
        key = (str(row["user_id"]), str(row["item_id"]))
        if row["split"] == "train" and float(row["rating"]) >= 4 and key[1] in index and key not in unique:
            counts[index[key[1]]] += 1
            unique.add(key)
    return counts


def stable_topk(scores, k, excluded=()):
    scores = np.asarray(scores, dtype=np.float64).copy()
    if len(excluded):
        scores[np.asarray(list(excluded), dtype=np.int64)] = -np.inf
    valid = np.flatnonzero(np.isfinite(scores))
    count = min(max(int(k), 0), len(valid))
    if count == 0:
        return np.empty(0, dtype=np.int64)
    if count < len(valid):
        threshold = np.partition(scores[valid], -count)[-count]
        valid = valid[scores[valid] >= threshold]
    return valid[np.lexsort((valid, -scores[valid]))][:count]


class HistoryRecommender(nn.Module):
    """A content-based BPR ranker usable for existing and new users alike."""

    def __init__(self, text, image=None, embedding_dim=128, max_history=20):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_history = max_history
        # Features are immutable external artifacts, not duplicated in checkpoints.
        self.register_buffer("text_features", F.normalize(torch.as_tensor(text), dim=-1), persistent=False)
        self.register_buffer("image_features", None if image is None else F.normalize(torch.as_tensor(image), dim=-1), persistent=False)
        self.text_projection = nn.Linear(text.shape[1], embedding_dim)
        self.image_projection = None if image is None else nn.Linear(image.shape[1], embedding_dim)
        width = embedding_dim if image is None else embedding_dim * 2
        self.fusion = nn.Sequential(nn.Linear(width, embedding_dim), nn.GELU(), nn.LayerNorm(embedding_dim))
        self.user_network = nn.Sequential(nn.Linear(embedding_dim, embedding_dim), nn.GELU(), nn.Linear(embedding_dim, embedding_dim))

    def item_vectors(self, indices=None):
        text = self.text_features if indices is None else self.text_features[indices]
        vector = self.text_projection(text)
        if self.image_projection is not None:
            image = self.image_features if indices is None else self.image_features[indices]
            vector = torch.cat((vector, self.image_projection(image)), dim=-1)
        return F.normalize(self.fusion(vector), dim=-1)

    def user_vectors(self, histories, item_vectors=None):
        mask = histories >= 0
        vectors = self.item_vectors(histories.clamp_min(0)) if item_vectors is None else item_vectors[histories.clamp_min(0)]
        average = (vectors * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1)
        return F.normalize(self.user_network(average), dim=-1)

    def forward(self, histories, positives, negatives):
        users = self.user_vectors(histories)
        positive = (users * self.item_vectors(positives)).sum(dim=-1)
        negative = (users * self.item_vectors(negatives)).sum(dim=-1)
        return -F.logsigmoid(5.0 * (positive - negative)).mean()


class MatrixFactorization(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim=64):
        super().__init__()
        self.users = nn.Embedding(n_users, embedding_dim)
        self.items = nn.Embedding(n_items, embedding_dim)
        nn.init.normal_(self.users.weight, std=0.05)
        nn.init.normal_(self.items.weight, std=0.05)

    def forward(self, users, positives, negatives):
        user = self.users(users)
        difference = (user * (self.items(positives) - self.items(negatives))).sum(-1)
        return -F.logsigmoid(difference).mean()


def mf_fold_in(item_vectors, histories, regularization=1.0):
    """Ridge fold-in uses past likes only; the same rule serves UI and evaluation."""
    values = np.asarray(item_vectors, dtype=np.float64)
    inverse = np.linalg.inv(values.T @ values + regularization * np.eye(values.shape[1]))
    sums = np.stack([values[list(history)].sum(axis=0) for history in histories])
    return (sums @ inverse).astype(np.float32)
