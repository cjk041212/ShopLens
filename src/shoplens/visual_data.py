"""Read-only visualization data from the exported Qwen recommendation towers.

The two-dimensional map always uses one fixed reference candidate tower. Its
coordinates are navigation aids, not recommendation scores or high-dimensional
similarities. Leave-one-out comparisons call the actual exported user network.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from .models import stable_topk


def _array(values):
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    return np.asarray(values, dtype=np.float64)


def _unit_vectors(values):
    values = _array(values)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Expected a finite vector matrix")
    lengths = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(lengths <= 0):
        raise ValueError("Cosine similarity needs nonzero vectors")
    return values / lengths


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or int(value) != value or int(value) < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _sample_indices(item_ids, size, seed):
    """Select by a fixed ID hash; unrelated interactions never change the sample."""
    size = _integer(size, "sample size")
    keys = [(hashlib.sha256(f"{seed}:{item_id}".encode()).digest(), str(item_id), i)
            for i, item_id in enumerate(item_ids)]
    return sorted(row[2] for row in sorted(keys)[:size])


def _history(scorer, liked_item_ids, allow_empty=False):
    selected = list(dict.fromkeys(str(item_id) for item_id in liked_item_ids))
    unknown = set(selected) - set(scorer.index)
    if unknown:
        raise ValueError(f"Unknown liked item IDs: {sorted(unknown)[:5]}")
    if not selected and not allow_empty:
        raise ValueError("Select at least one liked product first")
    # Match qwen_inference: the UI appends new likes, and the scorer takes newest first.
    excluded = [scorer.index[item_id] for item_id in selected]
    active = list(reversed(excluded))[:scorer.max_history]
    return selected, excluded, active


def _item_frame(scorer, indices):
    return pd.DataFrame([
        {"item_id": scorer.item_ids[index], "title": scorer.items[index].get("title", "")}
        for index in indices
    ], columns=["item_id", "title"])


def catalog_projection(reference_scorer, fit_size=2048, seed=42):
    """Fit a fixed PCA and transform the *whole* reference catalog only once.

    Changing likes, Top-K or the selected comparison model cannot change these
    coordinates. All models must use this same reference projection; independently
    trained tower dimensions are not assumed to align. PCA is fitted to a fixed
    ID sample, with signs oriented deterministically by the largest loading.
    """
    fit_size = _integer(fit_size, "fit_size", 1)
    vectors = _unit_vectors(reference_scorer.item_vectors)
    item_ids = list(reference_scorer.item_ids)
    if len(vectors) != len(item_ids) or not item_ids:
        raise ValueError("Reference vectors must match a nonempty catalog")
    fit_indices = _sample_indices(item_ids, fit_size, seed)
    fitted = vectors[fit_indices]
    center = fitted.mean(axis=0)
    centered = fitted - center
    covariance = centered.T @ centered / max(len(fitted) - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues, kind="stable")[::-1][:2]
    components = eigenvectors[:, order].T
    for component in components:
        if component[np.argmax(np.abs(component))] < 0:
            component *= -1
    coordinates = (vectors - center) @ components.T
    if coordinates.shape[1] < 2:
        coordinates = np.pad(coordinates, ((0, 0), (0, 2 - coordinates.shape[1])))
    total_variance = float(np.maximum(eigenvalues, 0).sum())
    ratios = np.maximum(eigenvalues[order], 0) / total_variance if total_variance else np.zeros(len(order))
    return {
        "item_ids": item_ids,
        "coordinates": coordinates,
        "explained_variance_ratio": np.pad(ratios, (0, 2 - len(ratios))).tolist(),
        "reference_model": getattr(reference_scorer, "mode", "reference"),
        "fit_item_ids": [item_ids[index] for index in fit_indices],
        "sample_seed": seed,
        "note": "Fixed reference candidate-tower PCA; 2D distances do not replace high-dimensional cosine or user-network scores.",
    }


def history_candidate_similarities(scorer, liked_item_ids, candidate_item_ids):
    """Normalized history-to-candidate cosine in the selected model's full space."""
    _, _, active = _history(scorer, liked_item_ids)
    candidate_ids = list(dict.fromkeys(str(item_id) for item_id in candidate_item_ids))
    unknown = set(candidate_ids) - set(scorer.index)
    if unknown:
        raise ValueError(f"Unknown candidate item IDs: {sorted(unknown)[:5]}")
    candidates = [scorer.index[item_id] for item_id in candidate_ids]
    history_vectors = _unit_vectors(scorer.history_item_vectors)[active]
    candidate_vectors = _unit_vectors(scorer.item_vectors)[candidates]
    cosine = np.clip(history_vectors @ candidate_vectors.T, -1, 1)
    history = _item_frame(scorer, active)
    candidate_frame = _item_frame(scorer, candidates)
    return {
        "matrix": pd.DataFrame(cosine, index=history["item_id"].tolist(), columns=candidate_ids),
        "history": history,
        "candidates": candidate_frame,
        "note": "Full-dimensional normalized tower cosine is input evidence, not a causal contribution or a decomposition of the user-network score.",
    }


def build_item_map(reference_scorer, scorer, liked_item_ids, k=10, sample_size=500,
                   seed=42, projection=None):
    """Fixed background plus selected items/actual Top-K; empty likes show no scores."""
    k = _integer(k, "k", 1)
    if list(reference_scorer.item_ids) != list(scorer.item_ids):
        raise ValueError("Reference and selected model catalogs differ")
    selected, excluded, active = _history(scorer, liked_item_ids, allow_empty=True)
    projection = catalog_projection(reference_scorer, seed=seed) if projection is None else projection
    if list(projection["item_ids"]) != list(scorer.item_ids):
        raise ValueError("Projection catalog differs from the selected model")
    coordinates = _array(projection["coordinates"])
    if coordinates.shape != (len(scorer.item_ids), 2) or not np.isfinite(coordinates).all():
        raise ValueError("Invalid projection coordinates")
    scores = _array(scorer.score_histories([active]))[0] if active else np.full(len(scorer.item_ids), np.nan)
    if active and (scores.shape != (len(scorer.item_ids),) or not np.isfinite(scores).all()):
        raise ValueError("Invalid user-network scores")
    top = stable_topk(scores, k, excluded) if active else np.empty(0, dtype=np.int64)
    ranks = {int(index): rank for rank, index in enumerate(top, 1)}
    indices = sorted(set(_sample_indices(scorer.item_ids, sample_size, seed)) | set(excluded) | set(top))
    cosines = (history_candidate_similarities(scorer, liked_item_ids, [scorer.item_ids[i] for i in indices])["matrix"].to_numpy()
               if active else np.empty((0, len(indices))))
    excluded_set, active_set = set(excluded), set(active)
    points = _item_frame(scorer, indices)
    points["catalog_index"] = indices
    points["x"], points["y"] = coordinates[indices, 0], coordinates[indices, 1]
    points["role"] = ["history" if i in excluded_set else "recommended" if i in ranks else "background" for i in indices]
    points["active_history"] = [i in active_set for i in indices]
    points["rank"] = pd.array([ranks.get(i) for i in indices], dtype="Int64")
    points["score"] = scores[indices]
    points["max_history_cosine"] = cosines.max(axis=0) if active else np.full(len(indices), np.nan)
    points["mean_history_cosine"] = cosines.mean(axis=0) if active else np.full(len(indices), np.nan)
    top_items = _item_frame(scorer, top)
    top_items["rank"] = np.arange(1, len(top) + 1)
    top_items["score"] = scores[top]
    return {
        "points": points, "top_items": top_items,
        "active_history_ids": [scorer.item_ids[index] for index in active],
        "selected_item_ids": selected, "catalog_size": len(scorer.item_ids),
        "background_sample_size": min(_integer(sample_size, "sample_size"), len(scorer.item_ids)),
        "projection": projection,
        "note": "Coordinates stay in the fixed reference tower; highlights, scores and high-dimensional cosines use the selected model. All selected products are excluded from recommendations.",
    }


def leave_one_out(scorer, liked_item_ids, k=10, target_item_ids=None):
    """Re-score after removing one item from the *current* capped history.

    Each comparison keeps the same eligible catalog: every originally selected
    item remains excluded. Histories are not backfilled from older selections.
    Scores and full eligible-catalog ranks come from score_histories, not from
    cosine heuristics. This is a model sensitivity check, not a causal estimate.
    """
    k = _integer(k, "k", 1)
    selected, excluded, active = _history(scorer, liked_item_ids)
    histories = [active] + ([active[:i] + active[i + 1:] for i in range(len(active))] if len(active) > 1 else [])
    scores = _array(scorer.score_histories(histories))
    if scores.shape != (len(histories), len(scorer.item_ids)) or not np.isfinite(scores).all():
        raise ValueError("Invalid user-network scores")
    rankings = [stable_topk(row, len(scorer.item_ids), excluded) for row in scores]
    rank_maps = [{int(index): rank for rank, index in enumerate(order, 1)} for order in rankings]
    base_top = rankings[0][:k]
    baseline = _item_frame(scorer, base_top)
    baseline["rank"] = np.arange(1, len(base_top) + 1)
    baseline["score"] = scores[0, base_top]
    target_ids = ([scorer.item_ids[i] for i in base_top] if target_item_ids is None
                  else list(dict.fromkeys(str(item_id) for item_id in target_item_ids)))
    unknown = set(target_ids) - set(scorer.index)
    if unknown:
        raise ValueError(f"Unknown target item IDs: {sorted(unknown)[:5]}")
    eligible = rankings[0]
    summary_rows, effect_rows = [], []
    for position, removed in enumerate(active if len(active) > 1 else []):
        without = scores[position + 1]
        alternative_top = rankings[position + 1][:k]
        overlap = len(set(base_top) & set(alternative_top))
        summary_rows.append({
            "removed_item_id": scorer.item_ids[removed],
            "title": scorer.items[removed].get("title", ""),
            "topk_overlap": overlap,
            "topk_overlap_ratio": overlap / len(base_top) if len(base_top) else None,
            "mean_abs_score_change": float(np.abs(without[eligible] - scores[0, eligible]).mean()) if len(eligible) else None,
            "remaining_history_size": len(active) - 1,
        })
        for item_id in target_ids:
            target = scorer.index[item_id]
            effect_rows.append({
                "removed_item_id": scorer.item_ids[removed], "target_item_id": item_id,
                "baseline_score": float(scores[0, target]), "without_score": float(without[target]),
                "score_delta": float(without[target] - scores[0, target]),
                "baseline_rank": rank_maps[0].get(target), "without_rank": rank_maps[position + 1].get(target),
                "eligible": target in rank_maps[0],
            })
    summary = pd.DataFrame(summary_rows, columns=["removed_item_id", "title", "topk_overlap", "topk_overlap_ratio", "mean_abs_score_change", "remaining_history_size"])
    effects = pd.DataFrame(effect_rows, columns=["removed_item_id", "target_item_id", "baseline_score", "without_score", "score_delta", "baseline_rank", "without_rank", "eligible"])
    for column in ("baseline_rank", "without_rank"):
        effects[column] = pd.array(effects[column], dtype="Int64")
    return {
        "available": len(active) > 1,
        "reason": "" if len(active) > 1 else "Removing the only active history item would leave an unsupported empty history.",
        "summary": summary, "effects": effects, "baseline": baseline,
        "active_history_ids": [scorer.item_ids[index] for index in active],
        "selected_item_ids": selected, "eligible_catalog_size": len(eligible),
        "note": "Exact exported user-network sensitivity after removing one current history item; no backfill, all original selections remain excluded. This is not a causal attribution.",
    }
