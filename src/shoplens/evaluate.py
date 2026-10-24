"""Exact full-catalog chronological evaluation with user-macro ranking metrics."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

from .inference import FittedScorer, available_models, load_scorer
from .models import MODES, TemporalExamples, digest, load_events, stable_topk


def ranking_metrics(rows):
    """Average target metrics within each user, then equally across users."""
    if not rows:
        return {"recall@10": None, "ndcg@10": None, "recall@20": None,
                "ndcg@20": None, "n_users": 0, "n_targets": 0}
    by_user = defaultdict(list)
    for row in rows:
        by_user[row["user_id"]].append(row)
    result = {key: float(np.mean([np.mean([row[key] for row in values]) for values in by_user.values()]))
              for key in ("recall@10", "ndcg@10", "recall@20", "ndcg@20")}
    result.update(n_users=len(by_user), n_targets=len(rows))
    return result


def evaluate_model(processed_dir, artifact_dir, model="multimodal", split="test",
                   seed=None, batch_size=128, low_interaction_threshold=5,
                   max_targets=None, write=True, device="cpu"):
    if split not in {"validation", "test"}:
        raise ValueError("Evaluate validation or test only")
    if batch_size < 1 or low_interaction_threshold < 0:
        raise ValueError("batch_size must be positive and interaction threshold nonnegative")
    processed_dir, artifact_dir = Path(processed_dir), Path(artifact_dir)
    scorer = (load_scorer(processed_dir, artifact_dir, model) if device == "cpu"
              else FittedScorer(processed_dir, artifact_dir, model, device=device))
    metadata = scorer.metadata
    if seed is not None and seed != metadata["seed"]:
        raise ValueError(f"Requested seed {seed} differs from trained checkpoint seed {metadata['seed']}")
    if metadata["events_sha256"] != digest(processed_dir / "events.jsonl"):
        raise ValueError("Prepared interactions changed since training; evaluation would use a different split")
    events = load_events(processed_dir)
    examples = TemporalExamples(events, scorer.item_ids, scorer.max_history).examples(split)
    eligible = len(examples)
    if max_targets is not None:
        if max_targets < 1:
            raise ValueError("max_targets must be positive")
        examples = examples[:max_targets]
    if not examples:
        raise ValueError(f"No eligible {split} targets with earlier positive history and an unseen target")
    counts = metadata["training_item_counts"]
    rows = []
    for offset in range(0, len(examples), batch_size):
        batch = examples[offset:offset + batch_size]
        scores = scorer.score_histories([row["history"] for row in batch])
        for example, values in zip(batch, scores):
            ranking = stable_topk(values, 20, example["seen"])
            match = np.flatnonzero(ranking == example["target"])
            rank = None if not len(match) else int(match[0]) + 1
            row = {"user_id": example["user_id"], "target": example["target"]}
            for k in (10, 20):
                hit = rank is not None and rank <= k
                row[f"recall@{k}"] = float(hit)
                row[f"ndcg@{k}"] = float(1.0 / np.log2(rank + 1)) if hit else 0.0
            rows.append(row)
    low = [row for row in rows if counts[row["target"]] <= low_interaction_threshold]
    high = [row for row in rows if counts[row["target"]] > low_interaction_threshold]
    result = {"model": model, "split": split, **ranking_metrics(rows), "seed": metadata["seed"],
              "selected_epoch": metadata.get("selected_epoch"),
              "training_configuration": {key: metadata.get(key) for key in
                                         ("epochs", "n_training_examples", "max_examples", "lr", "weight_decay")},
              "low_interaction": {"max_training_positive_users": low_interaction_threshold, **ranking_metrics(low)},
              "higher_interaction": {"min_training_positive_users": low_interaction_threshold + 1, **ranking_metrics(high)},
              "items_sha256": metadata["items_sha256"], "events_sha256": metadata["events_sha256"],
              "featurehash": metadata.get("features_sha256"),
              "selected_features_sha256": metadata.get("selected_features_sha256"),
              "data_audit": metadata.get("data_audit", {}),
              "features_metadata": metadata.get("features_metadata", {}),
              "run_protocol": {
                  "candidate_set": "Full prepared catalog, excluding all strictly earlier positive items",
                  "history": "At most max_history distinct most recent positive items with timestamp < target timestamp",
                  "max_history": scorer.max_history,
                  "evaluation": "Prequential: earlier split interactions may enter later histories; model weights remain frozen",
                  "averaging": "Within-user target mean, then macro-average over users",
                  "positive_feedback": "rating >= 4",
                  "repeat_targets": "Skipped when already present in earlier positive history",
                  "ties": "Score descending, stable sorted item_id ascending",
                  "eligible_targets": eligible,
                  "evaluation_subset": len(examples) < eligible,
                  "n_catalog_items": len(scorer.item_ids),
                  "device": device if model in {"text", "multimodal", "adapted"} else "cpu",
                  "mf_inference": metadata.get("inference") if model == "mf" else None,
              }}
    if write:
        result_path = artifact_dir / "results.json"
        existing = json.loads(result_path.read_text()) if result_path.exists() else {"results": []}
        retained = [row for row in existing.get("results", [])
                    if (row.get("model"), row.get("split"), row.get("seed")) != (model, split, metadata["seed"])]
        retained.append(result)
        existing["results"] = retained
        artifact_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", dir=artifact_dir, delete=False, encoding="utf-8") as stream:
            json.dump(existing, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
            temporary = stream.name
        os.replace(temporary, result_path)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--model", choices=(*MODES, "all"), default="multimodal")
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--seed", type=int, help="Assert that the loaded checkpoint has this seed")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--low-interaction-threshold", type=int, default=5)
    parser.add_argument("--max-targets", type=int, help="Optional diagnostic subset, explicitly recorded")
    args = vars(parser.parse_args(argv))
    mode = args.pop("model")
    for selected in available_models(args["artifact_dir"]) if mode == "all" else [mode]:
        result = evaluate_model(model=selected, **args)
        print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
