"""Shared, deterministic temporal cases for the three Qwen adaptation objectives.

Prepared baseline files are read only. Review anchors are matched by the complete
(user, item, timestamp) key, never by ASIN alone. Recommendation queries retain
the old strictly earlier-history protocol and never contain target review text.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .models import TemporalExamples, digest, load_events, load_items


@dataclass
class QwenData:
    items: list[dict[str, Any]]
    item_ids: list[str]
    texts: list[str]
    events: list[dict[str, Any]]
    training_examples: list[dict[str, Any]]
    validation_examples: list[dict[str, Any]]
    adaptation_examples: list[dict[str, Any]]
    train_positive_sets: dict[str, set[int]]
    audit: dict[str, Any]
    hashes: dict[str, str]
    sample_metadata: dict[str, Any]


def _json_digest(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _case_key(case: dict[str, Any], item_ids: list[str]) -> tuple[str, str, int]:
    return str(case["user_id"]), item_ids[int(case["target"])], int(case["timestamp"])


def _case_record(case: dict[str, Any]) -> dict[str, Any]:
    # Only original behavior fields and the exact matched review enter this hash.
    # Sets/dict iteration order and other mutable debugging fields cannot affect it.
    record = {key: case[key] for key in ("user_id", "target", "history", "timestamp")}
    if "review" in case:
        record["review"] = case["review"]
    return record


def _positive_sets(events: list[dict[str, Any]], index: dict[str, int]) -> dict[str, set[int]]:
    result: dict[str, set[int]] = {}
    for event in events:
        if event["split"] == "train" and float(event["rating"]) >= 4:
            result.setdefault(str(event["user_id"]), set()).add(index[str(event["item_id"])])
    return result


def _matched_reviews(path: Path, events: list[dict[str, Any]], item_ids: set[str],
                     train_end: int) -> tuple[dict[tuple[str, str, int], str], dict[str, int]]:
    train_events = {}
    for event in events:
        if event["split"] == "train":
            key = str(event["user_id"]), str(event["item_id"]), int(event["timestamp"])
            if key in train_events:
                raise ValueError("Duplicate prepared training event key")
            train_events[key] = event
    reviews = {}
    seen = {}
    counts = {"review_records": 0, "nonpositive_reviews": 0, "short_reviews": 0,
              "identical_duplicate_reviews": 0}
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            counts["review_records"] += 1
            record = json.loads(line)
            try:
                key = str(record["user_id"]), str(record["item_id"]), int(record["timestamp"])
                rating = float(record["rating"])
            except (KeyError, ValueError, TypeError) as exc:
                raise ValueError(f"Invalid training review fields at line {line_number}") from exc
            if key[2] > train_end:
                raise ValueError(f"Review exceeds training cutoff at line {line_number}")
            if key[1] not in item_ids or key not in train_events:
                raise ValueError(f"Review has no matching (user,item,timestamp) train event at line {line_number}")
            if not np.isfinite(rating) or rating != float(train_events[key]["rating"]):
                raise ValueError(f"Review rating does not match its training event at line {line_number}")
            text = record.get("text") or ""
            if not isinstance(text, str):
                raise ValueError(f"Review text must be a string at line {line_number}")
            text = text.strip()
            identity = rating, text
            if key in seen:
                if seen[key] != identity:
                    raise ValueError(f"Conflicting reviews for one temporal event at line {line_number}")
                counts["identical_duplicate_reviews"] += 1
                continue
            seen[key] = identity
            if rating < 4:
                counts["nonpositive_reviews"] += 1
            elif len(text.split()) < 5:
                counts["short_reviews"] += 1
            else:
                reviews[key] = text
    return reviews, counts


def build_data(root: str | Path, domain: str, seed: int, adaptation_limit: int = 30000,
               max_history: int = 20) -> QwenData:
    if domain not in {"baby", "sports", "clothing"}:
        raise ValueError(f"Unknown prepared domain: {domain}")
    if adaptation_limit < 1 or max_history < 1:
        raise ValueError("adaptation_limit and max_history must be positive")
    processed = Path(root) / "data/processed" / domain
    items, events = load_items(processed), load_events(processed)
    item_ids = [str(item["item_id"]) for item in items]
    if not item_ids:
        raise ValueError("Prepared catalog is empty")
    index = {item_id: position for position, item_id in enumerate(item_ids)}
    if any(str(event["item_id"]) not in index for event in events):
        raise ValueError("Prepared event refers to a missing catalog item")
    audit = json.loads((processed / "audit.json").read_text(encoding="utf-8"))
    try:
        train_end = int(audit["cutoffs"]["train_end"])
        validation_end = int(audit["cutoffs"]["validation_end"])
    except (KeyError, ValueError, TypeError) as exc:
        raise ValueError("Prepared audit must contain train_end and validation_end") from exc
    if train_end > validation_end:
        raise ValueError("Prepared cutoffs are out of order")
    for event in events:
        timestamp = int(event["timestamp"])
        expected = "train" if timestamp <= train_end else "validation" if timestamp <= validation_end else "test"
        if event["split"] != expected:
            raise ValueError("Prepared event split disagrees with audit cutoffs")
    positive_sets = _positive_sets(events, index)
    temporal = TemporalExamples(events, item_ids, max_history)
    # The same eligibility rule as baseline BPR: at least one train-unobserved
    # negative must exist. No validation/test labels enter these positive sets.
    training = [case for case in temporal.examples("train", include_seen=False)
                if len(positive_sets[case["user_id"]]) < len(item_ids)]
    validation = temporal.examples("validation")
    reviews, review_counts = _matched_reviews(processed / "reviews_train.jsonl", events,
                                              set(item_ids), train_end)
    eligible = []
    for case in training:
        key = _case_key(case, item_ids)
        if key not in reviews:
            continue
        if case["target"] in case["history"] or len(case["history"]) != len(set(case["history"])):
            raise ValueError("Temporal case contains its target or repeated history items")
        enriched = {**case, "history": list(case["history"]), "review": reviews[key]}
        enriched["case_id"] = _json_digest(list(key))
        eligible.append(enriched)
    if not eligible:
        raise ValueError("No eligible train behavioral cases have matching positive reviews of at least five words")
    # Draw and order once per pipeline seed; all objectives consume this exact
    # list. Repeated calls with the same files/config/seed return identical cases.
    order = np.random.default_rng(seed).permutation(len(eligible))[:adaptation_limit]
    sampled = [eligible[int(position)] for position in order]
    hashes = {
        "items_sha256": digest(processed / "items.jsonl"),
        "events_sha256": digest(processed / "events.jsonl"),
        "reviews_train_sha256": digest(processed / "reviews_train.jsonl"),
        "audit_sha256": digest(processed / "audit.json"),
        "training_cases_sha256": _json_digest([_case_record(case) for case in training]),
        "validation_cases_sha256": _json_digest([_case_record(case) for case in validation]),
        "adaptation_cases_sha256": _json_digest([_case_record(case) for case in sampled]),
    }
    texts = [str(item.get("text") or item.get("title") or "") for item in items]
    if any(not text.strip() for text in texts):
        raise ValueError("Qwen candidate encoding requires nonempty real item text")
    metadata = {"seed": int(seed), "adaptation_limit": adaptation_limit, "max_history": max_history,
                "full_training_cases": len(training), "eligible_review_cases": len(eligible),
                "sampled_adaptation_cases": len(sampled), "validation_cases": len(validation),
                "sampling": "numpy default_rng(seed).permutation over baseline-order eligible train cases",
                "review_match": "exact user_id,item_id,timestamp; train rating>=4; at least five whitespace words",
                "negative_labels": "full training-period positives only; no validation/test labels",
                "history": "strictly earlier positive events, last distinct max_history items; target excluded",
                **review_counts}
    return QwenData(items, item_ids, texts, events, training, validation, sampled,
                    positive_sets, audit, hashes, metadata)


def sample_negatives(examples: list[dict[str, Any]], positive_sets: dict[str, set[int]],
                     n_items: int, seed: int) -> np.ndarray:
    if n_items < 1:
        raise ValueError("n_items must be positive")
    rng = np.random.default_rng(seed)
    result = np.empty(len(examples), dtype=np.int64)
    blocked, dense_pools = {}, {}
    for position, case in enumerate(examples):
        user = str(case["user_id"])
        if user not in blocked:
            if user not in positive_sets:
                raise ValueError("Missing training-positive set for sampled user")
            positives = frozenset(int(value) for value in positive_sets[user])
            if any(value < 0 or value >= n_items for value in positives):
                raise ValueError("Training-positive index is outside the catalog")
            if len(positives) >= n_items:
                raise ValueError("No train-unobserved negative item exists for sampled user")
            blocked[user] = positives
        positives = blocked[user]
        if int(case["target"]) not in positives:
            raise ValueError("Sampled target is not a training-period positive")
        if len(positives) > n_items // 2:
            if user not in dense_pools:
                dense_pools[user] = np.array([value for value in range(n_items) if value not in positives], dtype=np.int64)
            result[position] = rng.choice(dense_pools[user])
        else:
            candidate = int(rng.integers(n_items))
            while candidate in positives:
                candidate = int(rng.integers(n_items))
            result[position] = candidate
    return result
