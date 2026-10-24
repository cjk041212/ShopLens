"""Transparent local shopping rules and rank fusion over existing embeddings.

This module never loads a model, calls a service, or changes formal experiments.
Studio scores combine within-channel rank percentiles; they are not probabilities
or the recommendation scores reported by the formal evaluation pipeline.
"""

from __future__ import annotations

import copy
import csv
import io
import json
import math
import re
from collections.abc import Mapping

import numpy as np

NEGATIVE_WEIGHT = .25
AVOID_SEMANTIC_WEIGHT = .20
AVOID_TERMS_WEIGHT = .20


def parse_brief(text):
    """Extract editable English query/avoid/budget fields with visible rules only."""
    text = str(text or "").strip()
    working = list(text)
    amount = r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
    budget_patterns = [
        rf"\b(?:under|below|less than|up to|at most|no more than|max(?:imum)?(?: price)?|budget(?: of| is|:)?)[ \t]*\$?[ \t]*(?P<amount>{amount})(?:[ \t]*(?:dollars?|usd))?\b",
        rf"\$[ \t]*(?P<amount>{amount})[ \t]*(?:budget|or less|maximum|max)\b",
    ]
    budgets, spans = [], []
    for pattern in budget_patterns:
        for match in re.finditer(pattern, text, flags=re.I):
            budgets.append(float(match.group("amount").replace(",", "")))
            spans.append(match.span())
    notes = ["Parsed with transparent keyword rules; review and edit these fields before searching."]
    if not budgets:
        # An otherwise unqualified dollar amount is treated as an editable ceiling.
        for match in re.finditer(rf"\$[ \t]*(?P<amount>{amount})", text):
            budgets.append(float(match.group("amount").replace(",", "")))
            spans.append(match.span())
        if budgets:
            notes.append("A dollar amount without a budget phrase was interpreted as a price ceiling; please confirm.")
    for start, end in spans:
        working[start:end] = " " * (end - start)
    max_price = min(budgets) if budgets else None
    if len(set(budgets)) > 1:
        notes.append("Multiple budget amounts were found; the lowest ceiling was selected. Please confirm.")
    query = "".join(working)
    markers = list(re.finditer(r"\b(?:I\s+)?(?:don['’]t\s+like|do\s+not\s+like|avoid|excluding|exclude|without|no|not)\b(?![-–]|\s+only\b)", query, flags=re.I))
    avoid_parts, removal_spans = [], []
    for index, marker in enumerate(markers):
        end = markers[index + 1].start() if index + 1 < len(markers) else len(query)
        tail = query[marker.end():end]
        stop = re.search(r"[;.!?\n]|,\s*(?=(?:I\b|prefer\b|want\b|need\b|looking\b|ideally\b|with\b))|\b(?:but|and)\s+(?=(?:I\s+)?(?:prefer|want|need|like)\b)", tail, flags=re.I)
        if stop:
            end = marker.end() + stop.start()
        phrase = query[marker.end():end].strip(" ,;:.-")
        phrase = re.sub(r"\s+(?:and|or)\s*$", "", phrase, flags=re.I).strip()
        if phrase:
            avoid_parts.append(phrase)
            removal_spans.append((marker.start(), end))
    visible_query = list(query)
    for start, end in removal_spans:
        visible_query[start:end] = " " * (end - start)
    query = re.sub(r"\s+", " ", "".join(visible_query)).strip(" ,;:.-")
    query = re.sub(r"\b(?:and|but)\s*$", "", query, flags=re.I).strip(" ,;:.-")
    avoid = "; ".join(dict.fromkeys(avoid_parts))
    tags = []
    if max_price is not None:
        tags.append(f"Budget ≤ ${max_price:g}")
    if avoid:
        tags.append("Avoidance specified")
    for pattern, label in ((r"\bhik(?:e|ing)\b|\btrail\b", "Hiking"),
                           (r"\brunn?(?:ing)?\b|\bjogging\b", "Running"),
                           (r"\btravel\b|\btrip\b", "Travel"),
                           (r"\byoga\b", "Yoga"),
                           (r"\bbaby\b|\binfant\b|\bstroller\b", "Baby care"),
                           (r"\bwork\b|\boffice\b", "Work"),
                           (r"\beveryday\b|\bdaily\b", "Everyday")):
        if re.search(pattern, query, flags=re.I):
            tags.append(label)
    return {"query": query, "avoid": avoid, "max_price": max_price, "tags": tags,
            "parser": "transparent_rules", "notes": notes}


def _array(values):
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    return np.asarray(values, dtype=np.float32)


def _matrix(values, count, name):
    if values is None:
        return None
    values = _array(values)
    if values.ndim != 2 or values.shape[0] != count or values.shape[1] < 1 or not np.isfinite(values).all():
        raise ValueError(f"{name} must be a finite matrix with one row per catalog product")
    lengths = np.linalg.norm(values, axis=1, keepdims=True)
    return np.divide(values, lengths, out=np.zeros_like(values), where=lengths > 0)


def _vector(values, matrix, name):
    if values is None:
        return None
    if matrix is None:
        raise ValueError(f"{name} requires the corresponding catalog vector matrix")
    vector = _array(values).reshape(-1)
    if vector.shape != (matrix.shape[1],) or not np.isfinite(vector).all():
        raise ValueError(f"{name} has an invalid dimension or nonfinite values")
    norm = float(np.linalg.norm(vector))
    if not norm:
        raise ValueError(f"{name} must be a nonzero vector")
    return vector / norm


def _percentiles(values):
    """Average tied ranks over the entire fixed catalog; all ties map to 0.5."""
    values = np.asarray(values, dtype=np.float64)
    count = len(values)
    if count < 2:
        return np.full(count, .5)
    order = np.argsort(values, kind="stable")
    ordered = values[order]
    boundaries = np.r_[0, np.flatnonzero(ordered[1:] != ordered[:-1]) + 1, count]
    average = (boundaries[:-1] + boundaries[1:] - 1) / (2 * (count - 1))
    result = np.empty(count, dtype=np.float64)
    result[order] = np.repeat(average, np.diff(boundaries))
    return result


def _cosine(matrix, vector):
    return None if matrix is None or vector is None else np.clip(matrix @ vector, -1, 1)


def _blend(text, image, text_weight):
    if text is None:
        return image
    if image is None:
        return text
    return text_weight * text + (1 - text_weight) * image


def _positive_similarity(matrix, indices):
    if matrix is None or not indices:
        return None
    centroid = matrix[indices].mean(axis=0)
    norm = float(np.linalg.norm(centroid))
    return _cosine(matrix, centroid / norm) if norm else None


def _negative_similarity(matrix, indices):
    if matrix is None or not indices:
        return None
    # A maximum keeps dissimilar negative preferences from cancelling each other.
    result = np.full(len(matrix), -1., dtype=np.float32)
    for start in range(0, len(indices), 32):
        result = np.maximum(result, (matrix @ matrix[indices[start:start + 32]].T).max(axis=1))
    return np.clip(result, -1, 1)


def _catalog(items):
    items = list(items)
    ids = [str(item["item_id"]) for item in items]
    if len(set(ids)) != len(ids):
        raise ValueError("Catalog product IDs must be unique")
    return items, ids, {item_id: index for index, item_id in enumerate(ids)}


def _selected(values, index, name):
    values = list(dict.fromkeys(str(value) for value in values))
    unknown = set(values) - set(index)
    if unknown:
        raise ValueError(f"Unknown {name} product IDs: {sorted(unknown)[:5]}")
    return [index[value] for value in values]


def _weight(value, name):
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be between 0 and 1")
    return value


def _price(item):
    metadata = item.get("metadata") or {}
    value = metadata.get("price", item.get("price")) if isinstance(metadata, Mapping) else item.get("price")
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if not re.fullmatch(r"\$?\s*(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?", value):
            return None
        value = value.replace("$", "").replace(",", "").strip()
    try:
        price = float(value)
    except (ValueError, TypeError):
        return None
    return price if math.isfinite(price) and price >= 0 else None



def _category_paths(item):
    metadata = item.get("metadata") or {}
    categories = metadata.get("categories", []) if isinstance(metadata, Mapping) else []
    if not isinstance(categories, (list, tuple)):
        return []
    paths = []
    for path in categories:
        if isinstance(path, (list, tuple)) and path and all(isinstance(part, str) and part.strip() for part in path):
            paths.append(" / ".join(path))
    return list(dict.fromkeys(paths))


def _terms(values):
    if isinstance(values, str):
        values = re.split(r"[,;]|\b(?:and|or)\b", values, flags=re.I)
    return list(dict.fromkeys(re.sub(r"\s+", " ", str(term)).strip().lower()
                              for term in values if str(term).strip()))


def _text(item):
    metadata = item.get("metadata") or {}
    fields = [str(item.get(key, "") or "") for key in ("title", "description", "text")]
    if isinstance(metadata, Mapping):
        fields.extend([str(metadata.get("brand", "") or ""), json.dumps(metadata.get("categories", []), ensure_ascii=False)])
    return "\n".join(fields)


def _term_hits(text, terms):
    return [term for term in terms if re.search(r"(?<!\w)" + re.escape(term) + r"(?!\w)", text, flags=re.I)]


def _personalization(values, ids):
    if values is None:
        return None
    if isinstance(values, Mapping):
        missing = set(ids) - set(values)
        if missing:
            raise ValueError("Personalization scores must cover the full catalog")
        values = [values[item_id] for item_id in ids]
    values = _array(values)
    if values.shape != (len(ids),) or not np.isfinite(values).all():
        raise ValueError("Personalization scores must be finite and cover the full catalog")
    return _percentiles(values)



def _retrieval_pool(text_cos, image_cos, size, count):
    if size is not None:
        if isinstance(size, bool) or int(size) != size or int(size) < 1:
            raise ValueError("candidate_pool_size must be a positive integer or None")
        size = int(size)
    similarity = text_cos if text_cos is not None else image_cos
    channel = "text_query" if text_cos is not None else "image_query" if image_cos is not None else "feedback_only"
    if size is None or similarity is None or size >= count:
        return np.ones(count, dtype=bool), "disabled" if size is None else channel
    # Thresholding retains every boundary tie; filters never expand this pool.
    threshold = np.partition(similarity, count - size)[count - size]
    return similarity >= threshold, channel


def rank_products(items, text_vectors, image_vectors, *, query_vector=None, image_vector=None,
                  positive_ids=(), negative_ids=(), hidden_ids=(), text_weight=.65,
                  feedback_weight=.30, avoid_vector=None, avoid_terms=(), max_price=None,
                  k=12, personalization_scores=None, candidate_pool_size=200, required_category=None):
    """Fuse fixed full-catalog percentiles, apply disclosed penalties, then filter.

    Query text/image percentiles blend using text_weight, renormalizing a missing
    modality to full weight. Positive centroids use independently normalized text
    and image spaces; their fused percentile and optional user-network percentile
    are averaged. Query and available feedback blend using feedback_weight.
    Negative feedback, semantic avoidance, and any explicit avoid-term hit deduct
    .25, .20, and .20 times their respective percentile/binary signal. Avoidance is
    soft. Positive/negative/hidden selections are hard exclusions. With a budget,
    unknown or excessive historical prices are excluded. Ties resolve by item ID.
    Text queries first admit their top candidate_pool_size text-cosine products
    (all cutoff ties retained); image-only queries use image cosine. Filtering and
    reranking stay in that fixed pool and may return fewer than k products. None
    disables the gate for diagnostics; feedback-only requests use the full catalog.
    The pool is a retrieval boundary, not a guarantee of semantic relevance.
    An explicitly selected required_category matches an original nested metadata
    path (joined with " / ") exactly, ignoring case; unknown paths do not match.
    """
    items, ids, index = _catalog(items)
    if isinstance(k, bool) or int(k) != k or int(k) < 0:
        raise ValueError("k must be a nonnegative integer")
    if not items or not k:
        return []
    if required_category is not None:
        if not isinstance(required_category, str):
            raise ValueError("required_category must be a source category path string or None")
    text_weight = _weight(text_weight, "text_weight")
    feedback_weight = _weight(feedback_weight, "feedback_weight")
    if max_price is not None:
        if isinstance(max_price, bool):
            raise ValueError("max_price must be a finite nonnegative amount")
        max_price = float(max_price)
        if not math.isfinite(max_price) or max_price < 0:
            raise ValueError("max_price must be a finite nonnegative amount")
    text = _matrix(text_vectors, len(items), "text_vectors")
    image = _matrix(image_vectors, len(items), "image_vectors")
    query = _vector(query_vector, text, "query_vector")
    image_query = _vector(image_vector, image, "image_vector")
    avoid = _vector(avoid_vector, text, "avoid_vector")
    positive = _selected(positive_ids, index, "positive")
    negative = _selected(negative_ids, index, "negative")
    hidden = _selected(hidden_ids, index, "hidden")
    if set(positive) & set(negative):
        raise ValueError("A product cannot be both positive and negative feedback")
    text_cos, image_cos = _cosine(text, query), _cosine(image, image_query)
    pool, pool_channel = _retrieval_pool(text_cos, image_cos, candidate_pool_size, len(items))
    pool_count = int(pool.sum())
    query_text = None if text_cos is None else _percentiles(text_cos)
    query_image = None if image_cos is None else _percentiles(image_cos)
    query_score = _blend(query_text, query_image, text_weight)
    positive_text_cos = _positive_similarity(text, positive)
    positive_image_cos = _positive_similarity(image, positive)
    positive_text = None if positive_text_cos is None else _percentiles(positive_text_cos)
    positive_image = None if positive_image_cos is None else _percentiles(positive_image_cos)
    positive_score = _blend(positive_text, positive_image, text_weight)
    personal = _personalization(personalization_scores, ids)
    feedback = (positive_score + personal) / 2 if positive_score is not None and personal is not None else positive_score if personal is None else personal
    if query_score is None and feedback is None:
        return []
    if query_score is None:
        base = feedback
    elif feedback is None:
        base = query_score
    else:
        base = (1 - feedback_weight) * query_score + feedback_weight * feedback
    negative_text_cos = _negative_similarity(text, negative)
    negative_image_cos = _negative_similarity(image, negative)
    negative_score = _blend(None if negative_text_cos is None else _percentiles(negative_text_cos),
                            None if negative_image_cos is None else _percentiles(negative_image_cos), text_weight)
    avoid_cos = _cosine(text, avoid)
    avoid_score = None if avoid_cos is None else _percentiles(avoid_cos)
    terms = _terms(avoid_terms)
    # Metadata can affect only admitted candidates. Keep all embedding scores and
    # percentiles global, while avoiding repeated text/JSON work outside the pool.
    admitted = np.flatnonzero(pool)
    catalog_text = {i: _text(items[i]) for i in admitted} if terms else {}
    hits = {i: _term_hits(catalog_text[i], terms) if terms else [] for i in admitted}
    negative_penalty = np.zeros(len(items)) if negative_score is None else NEGATIVE_WEIGHT * negative_score
    semantic_penalty = np.zeros(len(items)) if avoid_score is None else AVOID_SEMANTIC_WEIGHT * avoid_score
    terms_penalty = np.zeros(len(items))
    for i in admitted:
        if hits[i]:
            terms_penalty[i] = AVOID_TERMS_WEIGHT
    scores = base - negative_penalty - semantic_penalty - terms_penalty
    excluded = set(positive) | set(negative) | set(hidden)
    prices = {i: _price(items[i]) for i in admitted}
    category_paths = {i: _category_paths(items[i]) for i in admitted}
    eligible = [i for i in admitted if i not in excluded
                and (required_category is None or required_category.casefold() in {path.casefold() for path in category_paths[i]})
                and (max_price is None or (prices[i] is not None and prices[i] <= max_price))]
    ranking = sorted(eligible, key=lambda i: (-float(scores[i]), ids[i]))[:int(k)]
    def scalar(values, i):
        return None if values is None else float(values[i])
    results = []
    for rank, i in enumerate(ranking, 1):
        item = items[i]
        components = {
            "query_text_cosine": scalar(text_cos, i), "query_image_cosine": scalar(image_cos, i),
            "query_text_percentile": scalar(query_text, i), "query_image_percentile": scalar(query_image, i),
            "query_score": scalar(query_score, i), "positive_text_cosine": scalar(positive_text_cos, i),
            "positive_image_cosine": scalar(positive_image_cos, i), "positive_score": scalar(positive_score, i),
            "personalization_percentile": scalar(personal, i), "feedback_score": scalar(feedback, i),
            "negative_score": scalar(negative_score, i), "avoid_semantic_score": scalar(avoid_score, i),
            "avoid_term_hits": hits[i], "base_score": float(base[i]),
            "negative_penalty": float(negative_penalty[i]), "avoid_semantic_penalty": float(semantic_penalty[i]),
            "avoid_terms_penalty": float(terms_penalty[i]), "historical_price": prices[i],
            "text_weight": text_weight, "feedback_weight": feedback_weight,
            "pool_size": None if candidate_pool_size is None else int(candidate_pool_size), "pool_count": pool_count, "pool_channel": pool_channel,
            "required_category": required_category, "source_category_paths": category_paths[i],
        }
        evidence = [f"Catalog title: {item.get('title', ids[i])}"]
        evidence.extend(f"Source category: {path}" for path in category_paths[i])
        if not category_paths[i]:
            evidence.append("Source category is unknown.")
        metadata = item.get("metadata") or {}
        if isinstance(metadata, Mapping) and metadata.get("brand"):
            evidence.append(f"Catalog brand: {metadata['brand']}")
        original = str(item.get("description") or item.get("text") or "")
        if original:
            evidence.append(f"Catalog text: {original[:220]}{'…' if len(original) > 220 else ''}")
        evidence.append(f"Historical metadata price: {prices[i]:.2f}; not a live quote." if prices[i] is not None
                        else "Historical metadata price is unknown.")
        if text_cos is not None:
            evidence.append(f"Text-query cosine: {float(text_cos[i]):.4f}; within-catalog percentile: {float(query_text[i]):.1%}.")
        if image_cos is not None:
            evidence.append(f"Image-query cosine: {float(image_cos[i]):.4f}; within-catalog percentile: {float(query_image[i]):.1%}.")
        for term in hits[i]:
            match = re.search(r"(?<!\w)" + re.escape(term) + r"(?!\w)", catalog_text[i], flags=re.I)
            snippet = catalog_text[i][max(0, match.start() - 45):match.end() + 65].replace("\n", " ")
            evidence.append(f"Explicit avoid term '{term}' occurs in catalog text: \"{snippet}\"; a soft penalty was applied.")
        reasons = []
        if pool_channel in ("text_query", "image_query"):
            reasons.append(f"Retrieval admits the top {candidate_pool_size} {pool_channel.replace('_', '-')} candidates, including cutoff ties ({pool_count} products); Studio reranks only this fixed pool. This is not a relevance guarantee.")
        elif pool_channel == "disabled":
            reasons.append("The query retrieval gate is disabled for diagnostics; Studio reranks the full catalog.")
        else:
            reasons.append("With no query, feedback-only reranking uses the full catalog.")
        if query_score is not None:
            reasons.append("Text/image retrieval uses separate full-catalog rank percentiles.")
        if feedback is not None:
            reasons.append("Positive-feedback content and/or exported user-network percentiles contribute to the Studio score.")
        if negative_score is not None:
            reasons.append("Similarity to disliked products receives a separate soft penalty.")
        if avoid_score is not None or hits[i]:
            reasons.append("Avoidance uses disclosed semantic and/or literal-text soft penalties.")
        if max_price is not None:
            reasons.append("The known historical metadata price is within the confirmed budget.")
        if required_category is not None:
            reasons.append("The product matches the explicitly selected original catalog category path.")
        results.append({"item_id": ids[i], "title": str(item.get("title", ids[i])), "rank": rank,
                        "score": float(scores[i]), "components": components, "evidence": evidence, "reasons": reasons,
                        "score_kind": "studio_rank_percentile_fusion"})
    return results


def export_shortlist(items, ids, brief, *, recommendations=(), source=None):
    """Return original shortlist metadata plus explicit local provenance; no I/O."""
    items, catalog_ids, index = _catalog(items)
    selected = _selected(ids, index, "shortlist")
    ranked = {str(row["item_id"]): row for row in recommendations}
    exported = []
    for i in selected:
        product = copy.deepcopy(items[i])
        product["item_id"] = catalog_ids[i]
        product["historical_price"] = _price(items[i])
        product["price_note"] = "Historical metadata price; not a live quote. Currency follows the source catalog."
        product["original_text"] = str(items[i].get("text") or items[i].get("description") or "")
        if catalog_ids[i] in ranked:
            product["studio_recommendation"] = copy.deepcopy(ranked[catalog_ids[i]])
        exported.append(product)
    return {"format": "shoplens_studio_shortlist_v1", "brief": copy.deepcopy(brief),
            "source": copy.deepcopy(source) if source is not None else {"metadata": "Prepared local product catalog"},
            "selected_item_ids": [catalog_ids[i] for i in selected], "items": exported,
            "note": "Shortlist selections and Studio fusion scores are session exploration, not new formal evaluation results."}


def shortlist_json(items, ids, brief, *, recommendations=(), source=None):
    return json.dumps(export_shortlist(items, ids, brief, recommendations=recommendations, source=source),
                      ensure_ascii=False, indent=2, allow_nan=False)


def _csv_cell(value):
    # Escape exported text cells only; CSV quoting alone does not stop formulas.
    if isinstance(value, str) and value.lstrip(" \t\r\n").startswith(("=", "+", "-", "@")):
        return "\'" + value
    return value


def shortlist_csv(items, ids, brief, *, recommendations=(), source=None):
    export = export_shortlist(items, ids, brief, recommendations=recommendations, source=source)
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=["item_id", "title", "historical_price", "price_note", "brand",
                                                  "categories", "original_text", "image_url", "source", "brief",
                                                  "studio_rank", "studio_score", "studio_components", "studio_evidence"])
    writer.writeheader()
    for item in export["items"]:
        metadata = item.get("metadata") or {}
        recommendation = item.get("studio_recommendation", {})
        row = {"item_id": item["item_id"], "title": item.get("title", ""),
                         "historical_price": item["historical_price"], "price_note": item["price_note"],
                         "brand": metadata.get("brand", "") if isinstance(metadata, Mapping) else "",
                         "categories": json.dumps(metadata.get("categories", []), ensure_ascii=False) if isinstance(metadata, Mapping) else "[]",
                         "original_text": item["original_text"], "image_url": item.get("image_url", ""),
                         "source": json.dumps(export["source"], ensure_ascii=False), "brief": json.dumps(export["brief"], ensure_ascii=False),
                         "studio_rank": recommendation.get("rank", ""), "studio_score": recommendation.get("score", ""),
                         "studio_components": json.dumps(recommendation.get("components", {}), ensure_ascii=False, allow_nan=False),
                         "studio_evidence": json.dumps(recommendation.get("evidence", []), ensure_ascii=False, allow_nan=False)}
        writer.writerow({key: _csv_cell(value) for key, value in row.items()})
    return output.getvalue()
