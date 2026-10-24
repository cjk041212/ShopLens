"""Automatic, aggregate-only diagnostics of fixed recommendation checkpoints.

This is a separately declared seed42 test-user subset, not the original 27-run
benchmark. Training labels define popularity; temporal held-out feedback defines
ranking outcomes. Neither review text nor per-user records are exported.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

DOMAINS = ("baby", "sports", "clothing")
MODELS = ("qwen_frozen", "qwen_semantic", "qwen_preference")
HISTORY_CAPS = (1, 3, 5, 10, 20)
BUCKETS = ("0", "1-5", "6-20", "21+")
METRICS = ("recall@10", "ndcg@10", "recall@20", "ndcg@20")
COHORT_SEED = 20261025


def popularity_bucket(count):
    if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count < 0:
        raise ValueError("Training positive-user counts must be nonnegative integers")
    return "0" if count == 0 else "1-5" if count <= 5 else "6-20" if count <= 20 else "21+"


def select_user_cohort(examples, max_users=1000, seed=COHORT_SEED):
    if type(max_users) is not int or max_users < 1:
        raise ValueError("max_users must be a positive integer")
    users = sorted({row["user_id"] for row in examples})
    if not users:
        raise ValueError("No eligible held-out users")
    positions = np.random.default_rng(seed).permutation(len(users))[:max_users]
    selected = {users[int(index)] for index in positions}
    return [row for row in examples if row["user_id"] in selected], len(users)


def _macro(rows):
    by_user = defaultdict(list)
    for row in rows:
        by_user[row["user_id"]].append(row)
    result = {"n_users": len(by_user), "n_targets": len(rows)}
    for metric in METRICS:
        result[metric] = float(np.mean([np.mean([row[metric] for row in values])
                                      for values in by_user.values()])) if by_user else None
    return result


def _user_ndcg(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["user_id"]].append(row["ndcg@10"])
    return {user: float(np.mean(values)) for user, values in grouped.items()}


def paired_user_delta(after, before, *, repeats=1000, seed=COHORT_SEED):
    """Conditional-on-checkpoint uncertainty, resampling users rather than events."""
    if set(after) != set(before) or not after or type(repeats) is not int or repeats < 1:
        raise ValueError("Paired changes require identical nonempty user cohorts")
    differences = np.array([after[user] - before[user] for user in sorted(after)], dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.empty(repeats)
    for index in range(repeats):
        means[index] = differences[rng.integers(0, len(differences), len(differences))].mean()
    return {"mean": float(differences.mean()), "ci95_low": float(np.quantile(means, .025)),
            "ci95_high": float(np.quantile(means, .975)), "bootstrap_repeats": repeats,
            "scope": "Paired user bootstrap conditional on the fixed seed42 checkpoint and selected cohort; not training-seed uncertainty."}


def exposure_metrics(exposure, counts, items):
    exposure, counts = np.asarray(exposure, dtype=np.int64), np.asarray(counts, dtype=np.int64)
    if exposure.shape != counts.shape or exposure.ndim != 1 or len(exposure) != len(items) or np.any(exposure < 0):
        raise ValueError("Exposure and training counts must match the catalog")
    total = int(exposure.sum())
    if not total:
        raise ValueError("Exposure analysis requires actual recommendation slots")
    shares = exposure / total
    ordered = np.sort(exposure)
    n_items = len(exposure)
    gini = 2 * float(np.dot(np.arange(1, n_items + 1), ordered)) / (n_items * total) - (n_items + 1) / n_items
    bucket_names = np.array([popularity_bucket(value) for value in counts])
    groups = []
    for bucket in BUCKETS:
        mask = bucket_names == bucket
        slots = int(exposure[mask].sum())
        groups.append({"bucket": bucket, "catalog_items": int(mask.sum()), "catalog_share": float(mask.mean()),
                       "recommendation_slots": slots, "exposure_share": slots / total})
    top = sorted(range(n_items), key=lambda index: (-int(exposure[index]), str(items[index]["item_id"])))
    return {"recommendation_slots": total, "covered_items": int(np.count_nonzero(exposure)),
            "catalog_items": n_items, "coverage@10": float(np.count_nonzero(exposure) / n_items),
            "gini": float(max(0., min(1., gini))), "hhi": float(np.dot(shares, shares)),
            "mean_training_positive_users": float(np.dot(shares, counts)),
            "top_1pct_exposure_share": float(shares[top[:max(1, math.ceil(n_items * .01))]].sum()),
            "popularity_exposure": groups,
            "most_exposed_products": [{"item_id": str(items[index]["item_id"]),
                                       "title": str(items[index].get("title", ""))[:160],
                                       "recommendation_slots": int(exposure[index]),
                                       "training_positive_users": int(counts[index])}
                                      for index in top[:5] if exposure[index] > 0]}


def evaluate_checkpoint(scorer, examples, counts, *, batch_size=128, bootstrap_repeats=1000):
    """Reuse the fixed scorer; FULL earlier-positive exclusions survive every cap."""
    from .models import stable_topk

    if not examples or type(batch_size) is not int or batch_size < 1:
        raise ValueError("Need nonempty cases and a positive batch size")
    if scorer.max_history != 20 or len(counts) != len(scorer.item_ids):
        raise ValueError("Expected the original 20-item history/catalog schema")
    for row in examples:
        history, seen, target = row["history"], row["seen"], row["target"]
        if not history or len(history) > 20 or len(history) != len(set(history)) or not set(history).issubset(seen) or target in seen:
            raise ValueError("Invalid unseen-target case or full prior-item exclusion set")
        if any(type(index) is not int or not 0 <= index < len(counts) for index in [target, *history, *seen]):
            raise ValueError("Temporal case contains invalid catalog indices")
    by_cap, user_scores, full_rows = [], {}, None
    exposures = np.zeros(len(counts), dtype=np.int64)
    lengths = np.array([len(row["history"]) for row in examples])
    for cap in HISTORY_CAPS:
        metrics = []
        for start in range(0, len(examples), batch_size):
            batch = examples[start:start + batch_size]
            scores = np.asarray(scorer.score_histories([row["history"][:cap] for row in batch]))
            if scores.shape != (len(batch), len(counts)) or not np.isfinite(scores).all():
                raise ValueError("Scorer returned invalid full-catalog scores")
            for row, values in zip(batch, scores):
                ranking = stable_topk(values, 20, row["seen"])
                matches = np.flatnonzero(ranking == row["target"])
                rank = int(matches[0]) + 1 if len(matches) else None
                metric = {"user_id": row["user_id"], "target": row["target"]}
                for k in (10, 20):
                    hit = rank is not None and rank <= k
                    metric[f"recall@{k}"] = float(hit)
                    metric[f"ndcg@{k}"] = float(1 / np.log2(rank + 1)) if hit else 0.
                metrics.append(metric)
                if cap == 20:
                    np.add.at(exposures, ranking[:10], 1)
        by_cap.append({"history_cap": cap, **_macro(metrics),
                       "effective_history_mean": float(np.minimum(lengths, cap).mean()),
                       "changed_history_targets": int(np.count_nonzero(lengths > cap))})
        user_scores[cap] = _user_ndcg(metrics)
        if cap == 20:
            full_rows = metrics
    for record in by_cap:
        record["paired_ndcg_delta_vs_cap20"] = paired_user_delta(
            user_scores[record["history_cap"]], user_scores[20], repeats=bootstrap_repeats)
    cohorts = []
    for bucket in BUCKETS:
        selected = [row for row in full_rows if popularity_bucket(counts[row["target"]]) == bucket]
        cohorts.append({"bucket": bucket, **_macro(selected)})
    return by_cap, cohorts, exposure_metrics(exposures, counts, scorer.items)


def run_diagnostics(runtime_root, *, max_users=1000, batch_size=128, bootstrap_repeats=1000, progress=None):
    from .models import TemporalExamples, digest, load_events, load_items, training_counts
    from .qwen_inference import QwenScorer

    runtime_root = Path(runtime_root)
    domain_records, curves, cohorts, exposures, provenance = [], [], [], [], []
    for domain in DOMAINS:
        processed = runtime_root / "data/processed" / domain
        items, events = load_items(processed), load_events(processed)
        item_ids = [str(item["item_id"]) for item in items]
        audit = json.loads((processed / "audit.json").read_text())
        train_end, validation_end = int(audit["cutoffs"]["train_end"]), int(audit["cutoffs"]["validation_end"])
        if train_end > validation_end:
            raise ValueError("Invalid chronological cutoffs")
        for event in events:
            expected = "train" if int(event["timestamp"]) <= train_end else "validation" if int(event["timestamp"]) <= validation_end else "test"
            if event["split"] != expected:
                raise ValueError("Prepared event split differs from audited cutoffs")
        all_cases = TemporalExamples(events, item_ids, 20).examples("test")
        selected, eligible_users = select_user_cohort(all_cases, max_users)
        counts = training_counts(events, item_ids)
        lengths = Counter(len(row["history"]) for row in selected)
        domain_records.append({"domain": domain, "catalog_items": len(items), "eligible_users": eligible_users,
                               "eligible_targets": len(all_cases), "selected_users": len({row["user_id"] for row in selected}),
                               "selected_targets": len(selected), "evaluation_subset": len(selected) != len(all_cases),
                               "history_length_distribution": [{"length": key, "targets": value} for key, value in sorted(lengths.items())],
                               "catalog_popularity": [{"bucket": bucket, "items": sum(popularity_bucket(value) == bucket for value in counts)} for bucket in BUCKETS]})
        events_hash = digest(processed / "events.jsonl")
        source_paths = [processed / name for name in ("items.jsonl", "events.jsonl", "audit.json")]
        for mode in MODELS:
            unit = runtime_root / "artifacts/qwen" / domain / mode / "seed42"
            if not (unit / "complete.json").is_file():
                raise ValueError("A selected formal checkpoint is incomplete")
            scorer = QwenScorer(processed, unit, mode, device="cpu")
            metadata = scorer.metadata
            if metadata.get("seed") != 42 or metadata.get("events_sha256") != events_hash or not np.array_equal(metadata["training_item_counts"], counts):
                raise ValueError("Checkpoint training provenance differs from prepared inputs")
            if metadata.get("selected_epoch") not in (1, 2) or metadata.get("epochs") != 2:
                raise ValueError("Expected validation-selected original adaptation checkpoint")
            history, popularity, exposure = evaluate_checkpoint(scorer, selected, counts, batch_size=batch_size,
                                                              bootstrap_repeats=bootstrap_repeats)
            identity = {"domain": domain, "model": mode, "seed": 42}
            curves.extend({**identity, **row} for row in history)
            cohorts.extend({**identity, **row} for row in popularity)
            exposures.append({**identity, **exposure})
            source_paths.extend([unit / "serving.npz", unit / "models" / f"{mode}.json", unit / "models" / f"{mode}.pt"])
            if progress:
                progress({**identity, "selected_users": domain_records[-1]["selected_users"],
                          "selected_targets": len(selected), "full_history_ndcg@10": history[-1]["ndcg@10"]})
            del scorer
        provenance.extend({"path": str(path.relative_to(runtime_root)), "bytes": path.stat().st_size,
                           "sha256": digest(path)} for path in source_paths)
    module_path = Path(__file__)
    result = {"schema_version": 1, "complete": True, "kind": "held_out_recommendation_diagnostics",
              "protocol": {"checkpoint_seed": 42, "cohort_seed": COHORT_SEED, "max_users_per_domain": max_users,
                           "models": list(MODELS), "domains": list(DOMAINS), "history_caps": list(HISTORY_CAPS),
                           "split": "test", "device": "cpu", "batch_size": batch_size,
                           "cohort_selection": "Sort eligible user IDs; NumPy default_rng(cohort_seed).permutation, take at most max_users; retain every eligible target for those users. Same cases for every objective and cap.",
                           "candidate_set": "Full prepared catalog, excluding ALL strictly earlier positive items even when visible history is truncated.",
                           "history": "Most recent distinct rating>=4 items with timestamp strictly before the target; cap truncates only model input.",
                           "evaluation": "Prequential fixed weights: earlier validation/test feedback can enter a later history; same-time feedback cannot.",
                           "averaging": "Within-user target mean, then equal user macro-average; popularity cohorts recompute macro-average within their eligible users.",
                           "popularity": "Distinct rating>=4 TRAIN users per item; bins 0,1-5,6-20,21+.",
                           "exposure": "Cap20 Top10 slots counted per eligible target (event weighted); coverage denominator is the full prepared catalog, including zero-exposure items.",
                           "bootstrap_repeats": bootstrap_repeats,
                           "bootstrap": "Paired user resampling within each cap vs cap20 comparison; conditional on one checkpoint/cohort, not independent training repeats.",
                           "training_runs_added": 0, "manual_labels_added": 0},
              "domains": domain_records, "history_curves": curves, "popularity_cohorts": cohorts,
              "exposure": exposures, "provenance": {"inputs": provenance, "diagnostic_code_sha256": hashlib.sha256(module_path.read_bytes()).hexdigest()},
              "limitations": ["One predeclared seed42 checkpoint per objective/domain and at most1000 test users per domain by default; separate from the original27 full-test records.",
                              "Historical rating>=4 events are positive feedback, not natural-language query relevance or explicit negative judgments.",
                              "History truncation changes model input at inference; models are not retrained for each cap. The mean-pooling head has no positional history model.",
                              "Popularity/coverage/exposure concentration describe this fixed cohort and catalog; coverage is not satisfaction, and low-interaction is not strict cold start.",
                              "Bootstrap intervals condition on selected checkpoints/users and do not measure training-seed uncertainty or establish a general fine-tuning gain."]}
    validate_diagnostics(result)
    return result


def validate_diagnostics(result):
    """Reject partial/nonfinite aggregate outputs before the UI plots them."""
    if not isinstance(result, dict) or result.get("schema_version") != 1 or result.get("complete") is not True or result.get("kind") != "held_out_recommendation_diagnostics":
        raise ValueError("Incomplete or unsupported recommendation diagnostics")
    protocol = result.get("protocol", {})
    if protocol.get("checkpoint_seed") != 42 or protocol.get("history_caps") != list(HISTORY_CAPS) or protocol.get("models") != list(MODELS) or protocol.get("domains") != list(DOMAINS):
        raise ValueError("Unexpected diagnostic model/history protocol")
    domain_rows = result.get("domains", [])
    if len(domain_rows) != 3 or {row.get("domain") for row in domain_rows} != set(DOMAINS):
        raise ValueError("Incomplete diagnostic domain cohorts")
    domain_counts = {row["domain"]: row for row in domain_rows}
    for row in domain_rows:
        if any(type(row.get(key)) is not int or row[key] < 1 for key in ("catalog_items", "selected_users", "selected_targets")) or row["selected_users"] > row["selected_targets"]:
            raise ValueError("Invalid selected domain cohort counts")
    expected = {(domain, model) for domain in DOMAINS for model in MODELS}
    for key, extra, choices in (("history_curves", "history_cap", HISTORY_CAPS), ("popularity_cohorts", "bucket", BUCKETS), ("exposure", None, (None,))):
        rows = result.get(key)
        wanted = {(domain, model, choice) for domain, model in expected for choice in choices}
        if not isinstance(rows, list) or len(rows) != len(wanted):
            raise ValueError(f"Incomplete diagnostic rows: {key}")
        identities = [(row.get("domain"), row.get("model"), row.get(extra) if extra else None) for row in rows]
        if set(identities) != wanted or len(set(identities)) != len(rows) or any(row.get("seed") != 42 for row in rows):
            raise ValueError(f"Invalid diagnostic identities: {key}")
    def check(value):
        if isinstance(value, dict):
            if any(key in {"user_id", "review", "events", "history_item_ids", "runtime_root"} for key in value):
                raise ValueError("Private per-user diagnostic fields must not be exported")
            for child in value.values():
                check(child)
        elif isinstance(value, list):
            for child in value:
                check(child)
        elif isinstance(value, float) and not math.isfinite(value):
            raise ValueError("Nonfinite diagnostic value")
    check(result)
    for row in result["history_curves"] + result["popularity_cohorts"]:
        if type(row.get("n_users")) is not int or type(row.get("n_targets")) is not int or not 0 <= row["n_users"] <= row["n_targets"]:
            raise ValueError("Invalid diagnostic cohort counts")
        for metric in METRICS:
            value = row.get(metric)
            if value is None and row["n_targets"] == 0:
                continue
            if type(value) not in (int, float) or not 0 <= value <= 1:
                raise ValueError("Invalid diagnostic ranking metric")
    for row in result["history_curves"]:
        domain = domain_counts[row["domain"]]
        if row["n_users"] != domain["selected_users"] or row["n_targets"] != domain["selected_targets"]:
            raise ValueError("History caps must retain the same domain cohort")
    for domain, model in expected:
        group = [row for row in result["popularity_cohorts"] if (row["domain"], row["model"]) == (domain, model)]
        if sum(row["n_targets"] for row in group) != domain_counts[domain]["selected_targets"]:
            raise ValueError("Popularity cohorts must partition all selected targets")
    for row in result["exposure"]:
        domain = domain_counts[row["domain"]]
        if row.get("catalog_items") != domain["catalog_items"] or type(row.get("covered_items")) is not int or not 0 < row["covered_items"] <= domain["catalog_items"]:
            raise ValueError("Invalid exposure catalog coverage")
        for key in ("coverage@10", "gini", "hhi", "top_1pct_exposure_share"):
            if type(row.get(key)) not in (int, float) or not 0 <= row[key] <= 1:
                raise ValueError("Invalid exposure share or concentration")
        if not math.isclose(row["coverage@10"], row["covered_items"] / row["catalog_items"], abs_tol=1e-12):
            raise ValueError("Inconsistent exposure coverage")
        groups = row.get("popularity_exposure", [])
        if len(groups) != 4 or {group.get("bucket") for group in groups} != set(BUCKETS):
            raise ValueError("Incomplete exposure popularity buckets")
        if sum(group["recommendation_slots"] for group in groups) != row["recommendation_slots"] or sum(group["catalog_items"] for group in groups) != row["catalog_items"]:
            raise ValueError("Exposure popularity buckets must partition slots and catalog")
    return result


def load_recommendation_diagnostics(path):
    def reject_constant(value):
        raise ValueError(f"Nonfinite JSON constant: {value}")
    return validate_diagnostics(json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=reject_constant))
