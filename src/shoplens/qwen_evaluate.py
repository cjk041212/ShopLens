"""Formal full-catalog evaluation of the selected Qwen candidate checkpoint."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np

from .evaluate import ranking_metrics
from .models import TemporalExamples, digest, load_events, stable_topk
from .qwen_inference import QwenScorer


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        temporary = stream.name
    os.replace(temporary, path)


def save_result(artifact_dir, result):
    """Idempotently repair summaries after an interruption without rerunning test."""
    artifact_dir = Path(artifact_dir)
    _atomic_json(artifact_dir / "results.json", {"results": [result]})
    summary_path = artifact_dir.parent.parent / "results.json"
    existing = json.loads(summary_path.read_text()) if summary_path.is_file() else {"results": []}
    identity = (result["model"], result["split"], result["seed"])
    existing["results"] = [row for row in existing.get("results", [])
                           if (row.get("model"), row.get("split"), row.get("seed")) != identity] + [result]
    _atomic_json(summary_path, existing)


def evaluate_qwen(processed_dir, artifact_dir, model, seed, device="cuda", write=True):
    processed_dir, artifact_dir = Path(processed_dir), Path(artifact_dir)
    device = str(device)
    scorer = QwenScorer(processed_dir, artifact_dir, model, device=device)
    metadata = scorer.metadata
    if int(seed) != int(metadata["seed"]):
        raise ValueError("Requested seed differs from the selected Qwen checkpoint")
    if metadata["events_sha256"] != digest(processed_dir / "events.jsonl"):
        raise ValueError("Prepared interactions changed since Qwen training")
    if metadata["selected_epoch"] not in (1, 2) or metadata["epochs"] != 2:
        raise ValueError("Formal evaluation requires a selected adaptation epoch1 or epoch2")
    events = load_events(processed_dir)
    examples = TemporalExamples(events, scorer.item_ids, scorer.max_history).examples("test")
    if not examples:
        raise ValueError("No eligible unseen test targets with strictly earlier history")
    counts = metadata["training_item_counts"]
    if len(counts) != len(scorer.item_ids):
        raise ValueError("Training interaction counts do not match the catalog")
    rows = []
    for offset in range(0, len(examples), 128):
        batch = examples[offset:offset + 128]
        scores = scorer.score_histories([example["history"] for example in batch])
        for example, values in zip(batch, scores):
            ranking = stable_topk(values, 20, example["seen"])
            match = np.flatnonzero(ranking == example["target"])
            rank = int(match[0]) + 1 if len(match) else None
            row = {"user_id": example["user_id"], "target": example["target"]}
            for k in (10, 20):
                hit = rank is not None and rank <= k
                row[f"recall@{k}"] = float(hit)
                row[f"ndcg@{k}"] = float(1.0 / np.log2(rank + 1)) if hit else 0.0
            rows.append(row)
    low = [row for row in rows if counts[row["target"]] <= 5]
    high = [row for row in rows if counts[row["target"]] > 5]
    result = {"model": model, "split": "test", "seed": int(seed), **ranking_metrics(rows),
              "selected_epoch": metadata["selected_epoch"],
              "training_configuration": {key: metadata.get(key) for key in
                  ("epochs", "n_training_examples", "max_examples", "lr", "weight_decay", "model_id",
                   "revision", "objective", "warmup", "encoding_config", "lora_config", "config_sha256")},
              "low_interaction": {"max_training_positive_users": 5, **ranking_metrics(low)},
              "higher_interaction": {"min_training_positive_users": 6, **ranking_metrics(high)},
              "items_sha256": metadata["items_sha256"], "events_sha256": metadata["events_sha256"],
              "featurehash": metadata["base_features_sha256"],
              "selected_features_sha256": metadata["candidate_sha256"],
              "serving_sha256": metadata["serving_sha256"], "head_sha256": metadata["head_sha256"],
              "adapter_sha256": metadata.get("adapter_sha256"),
              "data_audit": metadata.get("data_audit", {}), "features_metadata": metadata.get("features_metadata", {}),
              "run_protocol": {
                  "candidate_set": "Full prepared catalog, excluding all strictly earlier positive items",
                  "history": "At most max_history distinct most recent positive items with timestamp < target timestamp",
                  "max_history": scorer.max_history,
                  "evaluation": "Prequential: earlier split interactions may enter later histories; model weights remain frozen",
                  "averaging": "Within-user target mean, then macro-average over users",
                  "positive_feedback": "rating >= 4",
                  "repeat_targets": "Skipped when already present in earlier positive history",
                  "ties": "Score descending, stable sorted item_id ascending",
                  "eligible_targets": len(examples), "evaluation_subset": False,
                  "n_catalog_items": len(scorer.item_ids), "device": device, "mf_inference": None,
                  "history_encoder": "Frozen base Qwen text and fixed CLIP image features in all three modes",
                  "candidate_encoder": model,
                  "checkpoint_selection": "Best full-catalog downstream validation NDCG@10 among adaptation epochs1 and2",
                  "adaptation_budget": "Up to30000 shared TRAIN cases with matching positive review; full-input common warmup",
              }}
    if write:
        save_result(artifact_dir, result)
    return result
