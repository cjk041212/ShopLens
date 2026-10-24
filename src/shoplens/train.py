"""Train transparent baselines and history-conditioned content recommenders."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import DataLoader, TensorDataset

from .models import (MODES, NEURAL_MODES, HistoryRecommender, MatrixFactorization,
                     TemporalExamples, digest, feature_content_digest, load_events, load_features, load_items,
                     pad_histories, resolve_device, seed_everything,
                     stable_topk, training_counts)


def validation_ndcg(recommender, mode, examples, device, max_history, batch_size=128):
    """Exact all-catalog validation; the test split is never consulted."""
    recommender.eval()
    by_user = defaultdict(list)
    with torch.inference_mode():
        item_vectors = (recommender.item_vectors() if mode in NEURAL_MODES
                        else recommender.items.weight.detach().cpu().numpy())
        if mode == "mf":
            values = item_vectors.astype(np.float64)
            inverse = np.linalg.inv(values.T @ values + np.eye(values.shape[1]))
        for offset in range(0, len(examples), batch_size):
            batch = examples[offset:offset + batch_size]
            histories = [row["history"] for row in batch]
            if mode in NEURAL_MODES:
                tensor = torch.from_numpy(pad_histories(histories, max_history)).to(device)
                users = recommender.user_vectors(tensor, item_vectors)
                scores = (users @ item_vectors.T).cpu().numpy()
            else:
                sums = np.stack([item_vectors[history].sum(axis=0) for history in histories])
                scores = sums @ inverse @ item_vectors.T
            for row, values in zip(batch, scores):
                ranking = stable_topk(values, 10, row["seen"])
                match = np.flatnonzero(ranking == row["target"])
                value = 0.0 if not len(match) else float(1.0 / np.log2(int(match[0]) + 2))
                by_user[row["user_id"]].append(value)
    return float(np.mean([np.mean(values) for values in by_user.values()]))


def train_model(processed_dir, artifact_dir, model="multimodal", seed=42, epochs=5,
                batch_size=256, lr=1e-3, weight_decay=1e-4, device="cpu",
                max_history=20, embedding_dim=128, max_examples=None):
    if model not in MODES:
        raise ValueError(f"Unknown model: {model}")
    if epochs < 1 or batch_size < 1 or max_history < 1:
        raise ValueError("epochs, batch_size and max_history must be positive")
    processed_dir, artifact_dir = Path(processed_dir), Path(artifact_dir)
    output = artifact_dir / "models"
    output.mkdir(parents=True, exist_ok=True)
    seed_everything(seed)
    items, events = load_items(processed_dir), load_events(processed_dir)
    item_ids = [str(item["item_id"]) for item in items]
    if not item_ids:
        raise ValueError("Catalog is empty")
    counts = training_counts(events, item_ids)
    if not counts.sum():
        raise ValueError("No training-period positive feedback (rating >= 4)")
    audit_path = processed_dir / "audit.json"
    feature_metadata_path = artifact_dir / "features.json"
    audit = json.loads(audit_path.read_text()) if audit_path.exists() else {}
    feature_metadata = json.loads(feature_metadata_path.read_text()) if feature_metadata_path.exists() else {}
    metadata = {
        "model": model, "seed": seed, "item_ids": item_ids,
        "max_history": max_history, "embedding_dim": embedding_dim,
        "items_sha256": digest(processed_dir / "items.jsonl"),
        "events_sha256": digest(processed_dir / "events.jsonl"),
        "n_items": len(items), "n_train_positive_pairs": int(counts.sum()),
        "training_item_counts": counts.tolist(),
        "data_audit": audit, "features_metadata": feature_metadata,
        "protocol": "Global prepared cutoffs; rating>=4; strictly earlier positive history; unseen-item ranking",
        "epochs": 0, "training_loss": [], "device": str(device),
    }
    if model == "popularity":
        np.savez_compressed(output / "popularity.npz", counts=counts)
    elif model == "tfidf":
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=30000,
                                     sublinear_tf=True, strip_accents="unicode")
        texts = [str(item.get("text") or item.get("title") or "") for item in items]
        if not any(text.strip() for text in texts):
            raise ValueError("TF-IDF requires nonempty catalog text")
        matrix = vectorizer.fit_transform(texts)
        joblib.dump({"vectorizer": vectorizer, "matrix": matrix}, output / "tfidf.joblib")
        metadata["text_fit_scope"] = "Unsupervised product metadata for the fixed candidate catalog; no reviews"
    else:
        torch_device = resolve_device(device)
        examples = TemporalExamples(events, item_ids, max_history).examples("train", include_seen=False)
        validation = TemporalExamples(events, item_ids, max_history).examples("validation")
        if not validation:
            raise ValueError("No eligible validation targets; prepare a valid chronological validation split before training")
        positives_by_user = defaultdict(set)
        item_index = {item_id: index for index, item_id in enumerate(item_ids)}
        for event in events:
            if event["split"] == "train" and float(event["rating"]) >= 4 and str(event["item_id"]) in item_index:
                positives_by_user[str(event["user_id"])].add(item_index[str(event["item_id"])])
        # Full training-period positives prevent sampling an observed training like
        # as a negative. No validation/test labels enter negative sampling.
        examples = [row for row in examples if len(positives_by_user[row["user_id"]]) < len(items)]
        if max_examples is not None:
            if max_examples < 1:
                raise ValueError("max_examples must be positive")
            order = np.random.default_rng(seed).permutation(len(examples))[:max_examples]
            examples = [examples[index] for index in order]
        if not examples:
            raise ValueError("No eligible training examples: need earlier positive history and unseen negative candidates")
        user_ids = sorted({row["user_id"] for row in examples})
        user_index = {user: index for index, user in enumerate(user_ids)}
        histories = torch.from_numpy(pad_histories([row["history"] for row in examples], max_history))
        targets = torch.tensor([row["target"] for row in examples], dtype=torch.long)
        users = torch.tensor([user_index[row["user_id"]] for row in examples], dtype=torch.long)
        if model in NEURAL_MODES:
            text, image = load_features(artifact_dir, item_ids, model)
            recommender = HistoryRecommender(text, image, embedding_dim, max_history).to(torch_device)
            metadata["features_sha256"] = digest(artifact_dir / "features.npz")
            metadata["selected_features_sha256"] = feature_content_digest(text, image)
            metadata["text_feature_dimension"] = int(text.shape[1])
            metadata["image_feature_dimension"] = None if image is None else int(image.shape[1])
        else:
            recommender = MatrixFactorization(len(user_ids), len(items), embedding_dim).to(torch_device)
            metadata["user_ids"] = user_ids
            metadata["inference"] = "Ridge history fold-in, lambda=1, shared by evaluation and new-user UI"
        optimizer = torch.optim.AdamW(recommender.parameters(), lr=lr, weight_decay=weight_decay)
        rng = np.random.default_rng(seed)
        # Never materialize user_count x catalog_size negative pools.
        dense_pools = {}
        def negative_for(user):
            positives = positives_by_user[user]
            if len(positives) > len(items) // 2:
                if user not in dense_pools:
                    dense_pools[user] = np.array([index for index in range(len(items)) if index not in positives])
                return int(rng.choice(dense_pools[user]))
            candidate = int(rng.integers(len(items)))
            while candidate in positives:
                candidate = int(rng.integers(len(items)))
            return candidate
        best_score, best_state, best_epoch = -1.0, None, None
        metadata["validation_ndcg@10"] = []
        for epoch in range(epochs):
            negatives = torch.tensor([negative_for(row["user_id"]) for row in examples], dtype=torch.long)
            dataset = TensorDataset(histories, users, targets, negatives)
            generator = torch.Generator().manual_seed(seed + epoch)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator, num_workers=0)
            recommender.train()
            total_loss, total = 0.0, 0
            for history, user, positive, negative in loader:
                optimizer.zero_grad(set_to_none=True)
                if model in NEURAL_MODES:
                    loss = recommender(history.to(torch_device), positive.to(torch_device), negative.to(torch_device))
                else:
                    loss = recommender(user.to(torch_device), positive.to(torch_device), negative.to(torch_device))
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite BPR training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(recommender.parameters(), 5.0)
                optimizer.step()
                total_loss += float(loss.detach().cpu()) * len(positive)
                total += len(positive)
            value = total_loss / total
            metadata["training_loss"].append(value)
            score = validation_ndcg(recommender, model, validation, torch_device, max_history)
            metadata["validation_ndcg@10"].append(score)
            if score > best_score:
                best_score, best_epoch = score, epoch + 1
                best_state = {key: tensor.detach().cpu().clone() for key, tensor in recommender.state_dict().items()}
            print(json.dumps({"model": model, "epoch": epoch + 1, "loss": value,
                              "validation_ndcg@10": score, "seed": seed}), flush=True)
        metadata["epochs"] = epochs
        metadata["n_training_examples"] = len(examples)
        metadata["max_examples"] = max_examples
        metadata["lr"] = lr
        metadata["weight_decay"] = weight_decay
        metadata["selected_epoch"] = best_epoch
        metadata["selection_metric"] = "validation user-macro ndcg@10, full candidate catalog"
        metadata["selected_validation_ndcg@10"] = best_score
        torch.save({"state_dict": best_state}, output / f"{model}.pt")
    path = output / f"{model}.json"
    path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--model", choices=(*MODES, "all"), default="multimodal")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--max-history", type=int, default=20)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--max-examples", type=int, help="Optional diagnostic subset; recorded in the model metadata")
    args = vars(parser.parse_args(argv))
    mode = args.pop("model")
    for selected in MODES if mode == "all" else [mode]:
        metadata = train_model(model=selected, **args)
        print(json.dumps({"saved": selected, "seed": metadata["seed"], "n_items": metadata["n_items"]}), flush=True)


if __name__ == "__main__":
    main()
