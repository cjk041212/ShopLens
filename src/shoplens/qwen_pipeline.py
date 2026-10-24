"""Resumable, matched-objective Qwen candidate-LoRA experiments."""
from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import random
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from .models import digest, pad_histories, resolve_device, seed_everything, training_counts
from .qwen_data import build_data, sample_negatives
from .qwen_models import QwenHistoryRecommender, export_serving, multi_positive_infonce, validate_full_catalog
from .qwen_text import (LORA_CONFIG, MODEL_ID, REVIEW_INSTRUCTION, REVISION, adapter_parameters,
                        adapter_state, embed_texts, encode_texts, load_text_model, restore_adapter,
                        review_query, save_adapter)

MODES = ("qwen_frozen", "qwen_semantic", "qwen_preference")
DEFAULT_CONFIG = {
    "max_length": 256, "max_history": 20, "embedding_dim": 128,
    "warmup_epochs": 20, "warmup_patience": 3, "warmup_min_epochs": 5,
    "warmup_batch_size": 256, "warmup_lr": 1e-3, "warmup_weight_decay": 1e-4,
    "adaptation_limit": 30000, "adaptation_epochs": 2, "adaptation_batch_size": 32,
    "head_lr": 1e-4, "head_weight_decay": 1e-4, "lora_lr": 2e-5, "lora_weight_decay": .01,
    "lora_config": LORA_CONFIG, "semantic_temperature": .07,
    "export_batch_size": 64, "checkpoint_steps": 100,
    "review_instruction": REVIEW_INSTRUCTION, "max_runtime_seconds": 0, "diagnostic": False,
}


class ResumableStop(RuntimeError):
    """Time window ended after a complete optimizer step and a saved recovery point."""


def _json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def _atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write("\n")
        temporary = stream.name
    os.replace(temporary, path)


def _atomic_torch(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = stream.name
    torch.save(value, temporary); os.replace(temporary, path)


def _atomic_npz(path, **values):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as stream:
        temporary = stream.name
    np.savez_compressed(temporary, **values); os.replace(temporary, path)


def _cpu_state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _state_hash(state):
    result = hashlib.sha256()
    for key, value in sorted(state.items()):
        result.update(key.encode()); result.update(str((value.shape, value.dtype)).encode())
        result.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return result.hexdigest()


def _tree_hash(directory):
    value = hashlib.sha256()
    for path in sorted(Path(directory).rglob("*")):
        if path.is_file():
            value.update(str(path.relative_to(directory)).encode()); value.update(digest(path).encode())
    return value.hexdigest()


def _rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def _restore_rng(state):
    random.setstate(state["python"]); np.random.set_state(state["numpy"]); torch.set_rng_state(state["torch"].cpu())
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def _deadline_reached(start, config):
    return config["max_runtime_seconds"] > 0 and time.monotonic() - start >= config["max_runtime_seconds"]


def _config(config):
    result = json.loads(json.dumps(DEFAULT_CONFIG))
    unknown = set(config or {}) - set(result)
    if unknown:
        raise ValueError(f"Unknown Qwen experiment config keys: {sorted(unknown)}")
    result.update(config or {})
    for key in ("max_length", "max_history", "embedding_dim", "warmup_epochs", "warmup_patience", "warmup_min_epochs",
                "warmup_batch_size", "adaptation_limit", "adaptation_epochs", "adaptation_batch_size", "export_batch_size", "checkpoint_steps"):
        if not isinstance(result[key], int) or result[key] < 1:
            raise ValueError(f"Positive integer required: {key}")
    if not result["diagnostic"]:
        fixed = {"max_length": 256, "max_history": 20, "embedding_dim": 128, "warmup_epochs": 20,
                 "warmup_min_epochs": 5, "warmup_patience": 3, "adaptation_epochs": 2, "adaptation_limit": 30000,
                 "warmup_batch_size": 256, "adaptation_batch_size": 32}
        if any(result[key] != value for key, value in fixed.items()) or result["lora_config"] != LORA_CONFIG:
            raise ValueError("Formal protocol constants changed; use diagnostic=True for a pilot")
    return result


def _base_features(root, domain, data, model_path, device, config):
    directory = root / "artifacts/qwen" / domain / "base"
    path, metadata_path = directory / "features.npz", directory / "features.json"
    source_path = root / "artifacts" / domain / "features.npz"
    source_meta = json.loads((source_path.parent / "features.json").read_text())
    if source_meta.get("features_sha256") != digest(source_path):
        raise ValueError("Frozen baseline CLIP feature checksum mismatch")
    identity = {"items_sha256": data.hashes["items_sha256"], "old_features_sha256": digest(source_path),
                "model_id": MODEL_ID, "revision": REVISION, "max_length": config["max_length"],
                "compute_dtype": "bfloat16" if str(device).startswith("cuda") else "float32", "diagnostic": config["diagnostic"]}
    if path.exists() or metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("identity") != identity or metadata.get("features_sha256") != digest(path):
            raise ValueError("Qwen base cache data/config/checksum changed; use a fresh experiment directory")
    else:
        with np.load(source_path, allow_pickle=False) as source:
            if source["item_ids"].astype(str).tolist() != data.item_ids:
                raise ValueError("CLIP rows do not match the prepared catalog")
            images = source["image"].astype(np.float32)
        model, tokenizer = load_text_model(model_path, device, trainable=False)
        text = encode_texts(model, tokenizer, data.texts, device, config["export_batch_size"], config["max_length"])
        del model, tokenizer; gc.collect()
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
        if not config["diagnostic"] and (text.shape != (len(data.item_ids), 1024) or images.shape != (len(data.item_ids), 512)):
            raise ValueError("Formal Qwen/CLIP feature dimensions must be 1024/512")
        _atomic_npz(path, item_ids=np.asarray(data.item_ids), text=text, image=images)
        metadata = {"identity": identity, "features_sha256": digest(path), "model_id": MODEL_ID, "revision": REVISION,
                    "synthetic": data.audit.get("synthetic"), "real_data": data.audit.get("synthetic") is False,
                    "encoding_config": {"max_length": config["max_length"], "padding_side": "left", "pooling": "last valid token", "normalization": "L2", "truncation": True},
                    "diagnostic": config["diagnostic"]}
        _atomic_json(metadata_path, metadata)
    with np.load(path, allow_pickle=False) as source:
        if source["item_ids"].astype(str).tolist() != data.item_ids:
            raise ValueError("Qwen base cache row order mismatch")
        text, images = source["text"].copy(), source["image"].copy()
    return text, images, metadata


def _make_ranker(base_text, image, config, device):
    return QwenHistoryRecommender(base_text, image, embedding_dim=config["embedding_dim"], max_history=config["max_history"]).to(device)


def _batch_arrays(examples, order, negatives, start, batch_size, device, max_history):
    positions = order[start:start + batch_size]
    selected = [examples[int(index)] for index in positions]
    histories = torch.as_tensor(pad_histories([row["history"] for row in selected], max_history), device=device)
    positives = torch.tensor([row["target"] for row in selected], dtype=torch.long, device=device)
    negative = torch.as_tensor(negatives[positions], device=device)
    return positions, histories, positives, negative


def _warmup(root, domain, seed, data, base_text, images, device, config, fingerprint, started):
    directory = root / "artifacts/qwen" / domain / "warmup" / f"seed{seed}"
    complete, latest = directory / "complete.json", directory / "resume.pt"
    model = _make_ranker(base_text, images, config, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["warmup_lr"], weight_decay=config["warmup_weight_decay"])
    if complete.exists():
        metadata = json.loads(complete.read_text())
        if metadata.get("fingerprint") != fingerprint or metadata.get("head_sha256") != digest(directory / "selected.pt"):
            raise ValueError("Warmup completion fingerprint/checksum changed")
        model.load_state_dict(torch.load(directory / "selected.pt", map_location=device, weights_only=True)["state_dict"])
        return model, metadata
    progress = {"epoch": 0, "next_start": 0, "total_loss": 0., "n_examples": 0,
                "curve": [], "best_score": -1., "best_epoch": None, "best_state": None, "stale": 0}
    if latest.exists():
        saved = torch.load(latest, map_location=device, weights_only=False)
        if saved["fingerprint"] != fingerprint or saved.get("stage") != "warmup":
            raise ValueError("Warmup resume data/config changed")
        model.load_state_dict(saved["head"]); optimizer.load_state_dict(saved["optimizer"])
        progress = saved["progress"]; _restore_rng(saved["rng"])
    examples = data.training_examples
    def save():
        _atomic_torch(latest, {"fingerprint": fingerprint, "stage": "warmup", "head": _cpu_state(model), "optimizer": optimizer.state_dict(),
                               "progress": progress, "rng": _rng_state()})
    stopped = progress["epoch"] >= config["warmup_min_epochs"] and progress["stale"] >= config["warmup_patience"]
    for epoch in range(config["warmup_epochs"] if stopped else progress["epoch"], config["warmup_epochs"]):
        order = np.random.default_rng(seed + 1000 + epoch).permutation(len(examples))
        negatives = sample_negatives(examples, data.train_positive_sets, len(data.items), seed + 2000 + epoch)
        model.train()
        for start in range(progress["next_start"], len(order), config["warmup_batch_size"]):
            if _deadline_reached(started, config):
                save(); raise ResumableStop(f"Saved {domain}/seed{seed} warmup before the time window ended")
            positions, histories, positives, negative = _batch_arrays(examples, order, negatives, start, config["warmup_batch_size"], device, config["max_history"])
            optimizer.zero_grad(set_to_none=True); loss = model(histories, positives, negative)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite warmup BPR loss")
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.); optimizer.step()
            progress["total_loss"] += float(loss.detach().cpu()) * len(positions); progress["n_examples"] += len(positions)
            progress["next_start"] = start + len(positions)
            if (start // config["warmup_batch_size"] + 1) % config["checkpoint_steps"] == 0:
                save()
                print(json.dumps({"stage": "warmup_progress", "domain": domain, "seed": seed, "epoch": epoch + 1,
                                  "next_start": progress["next_start"], "n_examples": progress["n_examples"],
                                  "elapsed_seconds": round(time.monotonic() - started, 2),
                                  "loss": progress["total_loss"] / progress["n_examples"]}), flush=True)
        save()
        if _deadline_reached(started, config):
            raise ResumableStop("Warmup trained epoch saved; validation will resume in the next window")
        score = validate_full_catalog(model, data.validation_examples)
        progress["curve"].append({"epoch": epoch + 1, "bpr_loss": progress["total_loss"] / progress["n_examples"], "validation_ndcg@10": score})
        if score > progress["best_score"]:
            progress.update(best_score=score, best_epoch=epoch + 1, best_state=_cpu_state(model), stale=0)
        else:
            progress["stale"] += 1
        print(json.dumps({"stage": "warmup", "domain": domain, "seed": seed, **progress["curve"][-1]}), flush=True)
        progress.update(epoch=epoch + 1, next_start=0, total_loss=0., n_examples=0)
        save()
        if epoch + 1 >= config["warmup_min_epochs"] and progress["stale"] >= config["warmup_patience"]:
            break
    model.load_state_dict(progress["best_state"])
    _atomic_torch(directory / "selected.pt", {"state_dict": _cpu_state(model)})
    metadata = {"fingerprint": fingerprint, "selected_epoch": progress["best_epoch"], "curve": progress["curve"],
                "head_sha256": digest(directory / "selected.pt"), "selection": "full-catalog validation NDCG@10",
                "max_epochs": config["warmup_epochs"], "patience": config["warmup_patience"], "min_epochs": config["warmup_min_epochs"]}
    _atomic_json(complete, metadata)
    return model, metadata


def _sampling(directory, data, seed, config, fingerprint):
    path, meta_path = directory / "sampling.npz", directory / "sampling.json"
    if path.exists() or meta_path.exists():
        metadata = json.loads(meta_path.read_text())
        if metadata.get("fingerprint") != fingerprint or metadata.get("sha256") != digest(path):
            raise ValueError("Shared adaptation sampling data/config changed")
    else:
        examples = data.adaptation_examples
        orders = np.stack([np.random.default_rng(seed + 3000 + epoch).permutation(len(examples)) for epoch in range(config["adaptation_epochs"])])
        negatives = np.stack([sample_negatives(examples, data.train_positive_sets, len(data.items), seed + 4000 + epoch) for epoch in range(config["adaptation_epochs"])])
        _atomic_npz(path, case_ids=np.asarray([row["case_id"] for row in examples]), orders=orders, negatives=negatives)
        _atomic_json(meta_path, {"fingerprint": fingerprint, "sha256": digest(path), "sample_metadata": data.sample_metadata})
    with np.load(path, allow_pickle=False) as values:
        if values["case_ids"].astype(str).tolist() != [row["case_id"] for row in data.adaptation_examples]:
            raise ValueError("Shared adaptation case order changed")
        return values["orders"].copy(), values["negatives"].copy(), digest(path)


def _review_anchors(directory, data, model_path, device, config, fingerprint):
    path, meta_path = directory / "review_anchors.npz", directory / "review_anchors.json"
    if path.exists() or meta_path.exists():
        info = json.loads(meta_path.read_text())
        if info.get("fingerprint") != fingerprint or info.get("sha256") != digest(path):
            raise ValueError("Review anchor cache changed")
    else:
        encoder, tokenizer = load_text_model(model_path, device, trainable=False)
        anchors = encode_texts(encoder, tokenizer, [review_query(row["review"], config["review_instruction"]) for row in data.adaptation_examples],
                               device, config["export_batch_size"], config["max_length"])
        _atomic_npz(path, case_ids=np.asarray([row["case_id"] for row in data.adaptation_examples]), anchors=anchors)
        _atomic_json(meta_path, {"fingerprint": fingerprint, "sha256": digest(path), "review_instruction": config["review_instruction"],
                                "source_reviews_sha256": data.hashes["reviews_train_sha256"]})
        del encoder, tokenizer; gc.collect()
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
    with np.load(path, allow_pickle=False) as values:
        if values["case_ids"].astype(str).tolist() != [row["case_id"] for row in data.adaptation_examples]:
            raise ValueError("Review anchor case order differs from behavior samples")
        return torch.as_tensor(values["anchors"].copy(), device=device)


def _norm(tensors):
    values = [tensor.detach().float().square().sum() for tensor in tensors if tensor is not None]
    return float(torch.stack(values).sum().sqrt().cpu()) if values else 0.


def _epoch_export(directory, epoch, ranker, encoder, tokenizer, data, device, config, fingerprint):
    destination = directory / "checkpoints" / f"epoch{epoch}"
    head_hash = _state_hash(_cpu_state(ranker))
    current_adapter = adapter_state(encoder) if encoder is not None else {}
    adapter_hash = _state_hash(current_adapter)
    manifest = destination / "epoch.json"
    if manifest.exists():
        previous = json.loads(manifest.read_text())
        if (previous["fingerprint"] != fingerprint or previous["head_tensor_sha256"] != head_hash
                or previous["adapter_tensor_sha256"] != adapter_hash):
            raise ValueError("Saved epoch export differs from the resumed training state")
        for name, checksum in previous["checksums"].items():
            if digest(destination / name) != checksum:
                raise ValueError("Saved epoch export checksum mismatch")
        with np.load(destination / "candidate.npz", allow_pickle=False) as values:
            ranker.set_candidate_text(values["text"])
        return previous
    candidate = (ranker.base_text.detach().cpu().numpy() if encoder is None
                 else encode_texts(encoder, tokenizer, data.texts, device, config["export_batch_size"], config["max_length"]))
    ranker.set_candidate_text(candidate)
    score = validate_full_catalog(ranker, data.validation_examples)
    serving = export_serving(ranker)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".epoch{epoch}-", dir=destination.parent))
    _atomic_npz(temporary / "candidate.npz", item_ids=np.asarray(data.item_ids), text=candidate)
    _atomic_npz(temporary / "serving.npz", item_ids=np.asarray(data.item_ids), **serving)
    _atomic_torch(temporary / "head.pt", {"state_dict": _cpu_state(ranker)})
    if encoder is not None:
        save_adapter(encoder, temporary / "adapter")
    # Probe live candidate text versus selected export and history/cache scores.
    probe = list(range(min(4, len(data.items))))
    was_training = ranker.training; ranker.eval()
    with torch.inference_mode():
        if encoder is not None:
            encoder.eval(); live = embed_texts(encoder, tokenizer, [data.texts[index] for index in probe], device, config["max_length"])
        else:
            live = ranker.base_text[probe]
        live_vectors = ranker.item_vectors(torch.tensor(probe, device=device), live)
        cached_vectors = torch.as_tensor(serving["candidate_item_vectors"][probe], device=device)
        maximum_error = float((live_vectors - cached_vectors).abs().max().cpu())
        histories = torch.as_tensor(pad_histories([data.validation_examples[0]["history"]], config["max_history"]), device=device)
        online_user = ranker.user_vectors(histories)
        cache_user = ranker.user_vectors(histories, torch.as_tensor(serving["history_item_vectors"], device=device))
        score_error = float(((online_user - cache_user) @ ranker.item_vectors().T).abs().max().cpu())
    ranker.train(was_training)
    tolerance = .02 if str(device).startswith("cuda") else 1e-5
    if maximum_error > tolerance or score_error > 1e-5:
        raise ValueError(f"Live/cache export parity failed: {maximum_error}, {score_error}")
    record = {"epoch": epoch, "validation_ndcg@10": score, "fingerprint": fingerprint,
              "head_tensor_sha256": head_hash, "adapter_tensor_sha256": adapter_hash,
              "checksums": {name: digest(temporary / name) for name in ("candidate.npz", "serving.npz", "head.pt")},
              "adapter_sha256": _tree_hash(temporary / "adapter") if encoder is not None else None,
              "live_cache_parity": {"candidate_max_abs_error": maximum_error, "history_score_max_abs_error": score_error,
                                     "candidate_tolerance": tolerance, "history_tolerance": 1e-5}}
    _atomic_json(temporary / "epoch.json", record)
    os.replace(temporary, destination)
    return record


def _finish_mode(root, domain, seed, mode, directory, progress, data, base_metadata, warmup, config, fingerprint, device, sampling_hash):
    selected = directory / "checkpoints" / f"epoch{progress['best_epoch']}"
    record = json.loads((selected / "epoch.json").read_text())
    (directory / "models").mkdir(parents=True, exist_ok=True)
    for source, target in (("candidate.npz", directory / "candidate.npz"), ("serving.npz", directory / "serving.npz"),
                           ("head.pt", directory / "models" / f"{mode}.pt")):
        shutil.copy2(selected / source, target)
    if (selected / "adapter").exists():
        shutil.copytree(selected / "adapter", directory / "adapter", dirs_exist_ok=True)
    metadata = {"model": mode, "seed": seed, "family": "qwen_candidate_lora", "item_ids": data.item_ids,
                "items_sha256": data.hashes["items_sha256"], "events_sha256": data.hashes["events_sha256"],
                "data_audit": data.audit, "training_item_counts": training_counts(data.events, data.item_ids).tolist(),
                "embedding_dim": config["embedding_dim"], "max_history": config["max_history"], "selected_epoch": progress["best_epoch"],
                "epochs": config["adaptation_epochs"], "max_examples": config["adaptation_limit"], "n_training_examples": len(data.adaptation_examples),
                "lr": config["head_lr"], "weight_decay": config["head_weight_decay"], "features_metadata": base_metadata,
                "serving_sha256": digest(directory / "serving.npz"), "head_sha256": digest(directory / "models" / f"{mode}.pt"),
                "candidate_sha256": digest(directory / "candidate.npz"), "base_features_sha256": base_metadata["features_sha256"],
                "model_id": MODEL_ID, "revision": REVISION, "objective": mode, "warmup": warmup,
                "encoding_config": {**base_metadata["encoding_config"], "review_instruction": config["review_instruction"]},
                "lora_config": None if mode == "qwen_frozen" else config["lora_config"],
                "adapter_sha256": record["adapter_sha256"], "config_sha256": fingerprint, "diagnostic": config["diagnostic"]}
    _atomic_json(directory / "models" / f"{mode}.json", metadata)
    training = {"model": mode, "seed": seed, "fingerprint": fingerprint, "config": config, "data_hashes": data.hashes,
                "warmup": warmup, "adaptation_curve": progress["curve"], "selected_epoch": progress["best_epoch"],
                "sample_metadata": data.sample_metadata, "sampling_sha256": sampling_hash,
                "gradient_evidence": progress["evidence"], "live_cache_parity": record["live_cache_parity"],
                "selected_adapter_sha256": record["adapter_sha256"], "selected_candidate_sha256": metadata["candidate_sha256"],
                "selected_serving_sha256": metadata["serving_sha256"], "diagnostic": config["diagnostic"]}
    training["gradient_evidence"]["selected_adapter_tensor_sha256"] = record["adapter_tensor_sha256"]
    training["gradient_evidence"]["selected_adapter_changed"] = (record["adapter_tensor_sha256"] != progress["evidence"]["initial_adapter_tensor_sha256"])
    with np.load(root / "artifacts/qwen" / domain / "warmup" / f"seed{seed}/sampling.npz", allow_pickle=False) as sampling:
        training["negative_sha256_by_epoch"] = [hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest() for values in sampling["negatives"]]
        training["case_order_sha256_by_epoch"] = [hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest() for values in sampling["orders"]]
    training["sample_hashes"] = {**data.hashes, "sampling_sha256": sampling_hash}
    training["adaptation_validation_ndcg@10"] = [row["validation_ndcg@10"] for row in progress["curve"]]
    training["model_id"], training["revision"] = MODEL_ID, REVISION
    training["encoding_config"], training["lora_config"] = metadata["encoding_config"], metadata["lora_config"]
    training["training_numerics"] = {
        "candidate_training_sdpa_backend": "not_applicable" if mode == "qwen_frozen" else
            ("math" if str(device).startswith("cuda") else "automatic"),
        "nonfinite_gradient_guard": True, "cached_encoding_sdpa_backend": "automatic"}
    _atomic_json(directory / "training.json", training)
    if config["diagnostic"]:
        result = None
        _atomic_json(directory / "diagnostic_complete.json", {"fingerprint": fingerprint, "diagnostic": True, "model": mode, "seed": seed,
                     "head_sha256": metadata["head_sha256"], "serving_sha256": metadata["serving_sha256"], "candidate_sha256": metadata["candidate_sha256"]})
    else:
        from .qwen_evaluate import evaluate_qwen, save_result
        # If an interruption happened after the atomic test result but before the
        # completion marker, reuse that exact result rather than evaluate twice.
        result_path = directory / "results.json"
        existing = json.loads(result_path.read_text()).get("results", []) if result_path.exists() else []
        if existing:
            result = existing[0]
            if (len(existing) != 1 or result.get("model") != mode or result.get("seed") != seed
                    or result.get("head_sha256") != metadata["head_sha256"] or result.get("serving_sha256") != metadata["serving_sha256"]):
                raise ValueError("Existing formal result belongs to another selected export")
            save_result(directory, result)
        else:
            result = evaluate_qwen(root / "data/processed" / domain, directory, mode, seed, device=device, write=True)
        _atomic_json(directory / "complete.json", {"fingerprint": fingerprint, "model": mode, "seed": seed,
                     "head_sha256": metadata["head_sha256"], "serving_sha256": metadata["serving_sha256"],
                     "candidate_sha256": metadata["candidate_sha256"], "adapter_sha256": metadata["adapter_sha256"],
                     "results_sha256": digest(result_path), "selected_epoch": progress["best_epoch"]})
    return {"model": mode, "seed": seed, "selected_epoch": progress["best_epoch"], "diagnostic": config["diagnostic"], "result": result}


def _train_mode(root, domain, seed, mode, data, base_text, images, base_metadata, warmup_model, warmup,
                orders, negatives, anchors, model_path, device, config, fingerprint, started, sampling_hash):
    directory = root / "artifacts/qwen" / domain / mode / f"seed{seed}"
    complete = directory / ("diagnostic_complete.json" if config["diagnostic"] else "complete.json")
    if complete.exists():
        previous = json.loads(complete.read_text())
        if previous.get("fingerprint") != fingerprint or previous.get("model") != mode or previous.get("seed") != seed:
            raise ValueError("Completed Qwen unit has different data/config")
        for key, path in (("head_sha256", directory / "models" / f"{mode}.pt"), ("serving_sha256", directory / "serving.npz"),
                          ("candidate_sha256", directory / "candidate.npz")):
            if previous.get(key) != digest(path):
                raise ValueError("Completed Qwen unit artifact checksum mismatch")
        if not config["diagnostic"]:
            if previous.get("results_sha256") != digest(directory / "results.json"):
                raise ValueError("Completed Qwen result checksum mismatch")
            if previous.get("adapter_sha256") is not None and previous["adapter_sha256"] != _tree_hash(directory / "adapter"):
                raise ValueError("Completed Qwen adapter checksum mismatch")
            from .qwen_evaluate import save_result
            saved_results = json.loads((directory / "results.json").read_text())["results"]
            if len(saved_results) != 1:
                raise ValueError("Completed unit must contain exactly one formal test result")
            save_result(directory, saved_results[0])
        return {"model": mode, "seed": seed, "skipped_verified_complete": True}
    seed_everything(seed)
    ranker = _make_ranker(base_text, images, config, device); ranker.load_state_dict(warmup_model.state_dict())
    encoder = tokenizer = None
    parameters = []
    if mode != "qwen_frozen":
        encoder, tokenizer = load_text_model(model_path, device, create_lora=True, lora_config=config["lora_config"])
        parameters = adapter_parameters(encoder)
        if not parameters or any(parameter.requires_grad for name, parameter in encoder.named_parameters() if "lora_" not in name):
            raise ValueError("Candidate tower must train LoRA parameters only")
    head_parameters = list(ranker.parameters())
    groups = [{"params": head_parameters, "lr": config["head_lr"], "weight_decay": config["head_weight_decay"]}]
    if parameters:
        groups.append({"params": parameters, "lr": config["lora_lr"], "weight_decay": config["lora_weight_decay"]})
    optimizer = torch.optim.AdamW(groups)
    latest = directory / "resume.pt"
    progress = {"epoch": 0, "next_start": 0, "bpr_sum": 0., "semantic_sum": 0., "n_examples": 0,
                "curve": [], "best_score": -1., "best_epoch": None,
                "evidence": {"base_parameters_frozen": True, "trainable_lora_parameters": sum(parameter.numel() for parameter in parameters),
                             "bpr_to_lora_max_norm": 0., "total_lora_gradient_max_norm": 0., "head_gradient_max_norm": 0., "gradient_probes": 0,
                             "initial_adapter_tensor_sha256": _state_hash(adapter_state(encoder) if encoder is not None else {})}}
    if latest.exists():
        saved = torch.load(latest, map_location=device, weights_only=False)
        if saved["fingerprint"] != fingerprint or saved.get("stage") != mode:
            raise ValueError("Adaptation resume data/config changed")
        ranker.load_state_dict(saved["head"])
        if encoder is not None:
            restore_adapter(encoder, saved["adapter"])
        optimizer.load_state_dict(saved["optimizer"]); progress = saved["progress"]; _restore_rng(saved["rng"])
    def save():
        _atomic_torch(latest, {"fingerprint": fingerprint, "stage": mode, "head": _cpu_state(ranker),
                               "adapter": adapter_state(encoder) if encoder is not None else {},
                               "optimizer": optimizer.state_dict(), "progress": progress, "rng": _rng_state()})
    examples = data.adaptation_examples
    for epoch in range(progress["epoch"], config["adaptation_epochs"]):
        ranker.train()
        if encoder is not None:
            encoder.train()
        for start in range(progress["next_start"], len(examples), config["adaptation_batch_size"]):
            if _deadline_reached(started, config):
                save(); raise ResumableStop(f"Saved {domain}/{mode}/seed{seed} for the next allocated window")
            positions, histories, positive, negative = _batch_arrays(examples, orders[epoch], negatives[epoch], start,
                          config["adaptation_batch_size"], device, config["max_history"])
            optimizer.zero_grad(set_to_none=True)
            semantic_loss = torch.zeros((), device=device)
            if encoder is None:
                bpr_loss = ranker(histories, positive, negative)
            else:
                concatenated = torch.cat((positive, negative)); unique, inverse = torch.unique(concatenated, sorted=True, return_inverse=True)
                live = embed_texts(encoder, tokenizer, [data.texts[int(index)] for index in unique.cpu()], device, config["max_length"])
                positive_text, negative_text = live[inverse[:len(positions)]], live[inverse[len(positions):]]
                if mode == "qwen_semantic":
                    bpr_loss = ranker(histories, positive, negative, positive_text.detach(), negative_text.detach())
                    semantic_loss = multi_positive_infonce(anchors[torch.as_tensor(positions, device=device)], positive_text, positive, config["semantic_temperature"])
                else:
                    bpr_loss = ranker(histories, positive, negative, positive_text, negative_text)
            loss = bpr_loss + semantic_loss
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite adaptation loss")
            evidence = progress["evidence"]
            probe = encoder is not None and evidence["gradient_probes"] < 10 and (evidence["gradient_probes"] == 0 or
                       evidence["total_lora_gradient_max_norm"] == 0 or (mode == "qwen_preference" and evidence["bpr_to_lora_max_norm"] == 0))
            if probe:
                gradients = torch.autograd.grad(bpr_loss, parameters, retain_graph=True, allow_unused=True)
                evidence["bpr_to_lora_max_norm"] = max(evidence["bpr_to_lora_max_norm"], _norm(gradients))
                evidence["gradient_probes"] += 1
            loss.backward()
            if probe or (encoder is None and evidence["head_gradient_max_norm"] == 0):
                evidence["total_lora_gradient_max_norm"] = max(evidence["total_lora_gradient_max_norm"], _norm([parameter.grad for parameter in parameters]))
                evidence["head_gradient_max_norm"] = max(evidence["head_gradient_max_norm"], _norm([parameter.grad for parameter in head_parameters]))
            torch.nn.utils.clip_grad_norm_(head_parameters, 5., error_if_nonfinite=True)
            if parameters:
                torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
            optimizer.step()
            progress["bpr_sum"] += float(bpr_loss.detach().cpu()) * len(positions)
            progress["semantic_sum"] += float(semantic_loss.detach().cpu()) * len(positions)
            progress["n_examples"] += len(positions); progress["next_start"] = start + len(positions)
            if (start // config["adaptation_batch_size"] + 1) % config["checkpoint_steps"] == 0:
                save()
                print(json.dumps({"stage": "adaptation_progress", "model": mode, "domain": domain, "seed": seed,
                                  "epoch": epoch + 1, "next_start": progress["next_start"], "n_examples": progress["n_examples"],
                                  "elapsed_seconds": round(time.monotonic() - started, 2),
                                  "bpr_loss": progress["bpr_sum"] / progress["n_examples"],
                                  "semantic_loss": progress["semantic_sum"] / progress["n_examples"]}), flush=True)
        save()
        if _deadline_reached(started, config):
            raise ResumableStop("Adaptation epoch trained and saved; full-catalog validation will resume")
        snapshot = _epoch_export(directory, epoch + 1, ranker, encoder, tokenizer, data, device, config, fingerprint)
        curve = {"epoch": epoch + 1, "bpr_loss": progress["bpr_sum"] / progress["n_examples"],
                 "semantic_loss": progress["semantic_sum"] / progress["n_examples"],
                 "validation_ndcg@10": snapshot["validation_ndcg@10"]}
        progress["curve"].append(curve)
        if snapshot["validation_ndcg@10"] > progress["best_score"]:
            progress.update(best_score=snapshot["validation_ndcg@10"], best_epoch=epoch + 1)
        print(json.dumps({"stage": mode, "domain": domain, "seed": seed, **curve}), flush=True)
        progress.update(epoch=epoch + 1, next_start=0, bpr_sum=0., semantic_sum=0., n_examples=0)
        save()
    if mode == "qwen_preference" and progress["evidence"]["bpr_to_lora_max_norm"] <= 0:
        raise ValueError("Preference loss never produced a nonzero LoRA gradient")
    if mode == "qwen_semantic" and (progress["evidence"]["bpr_to_lora_max_norm"] != 0 or progress["evidence"]["total_lora_gradient_max_norm"] <= 0):
        raise ValueError("Semantic gradient isolation did not hold")
    result = _finish_mode(root, domain, seed, mode, directory, progress, data, base_metadata, warmup, config, fingerprint, device, sampling_hash)
    del ranker, optimizer, encoder, tokenizer; gc.collect()
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return result


def run_one(root, domain, seed, model_path, device="cuda", config=None):
    root, device = Path(root).resolve(), resolve_device(device)
    config = _config(config); started = time.monotonic()
    seed_everything(seed)
    data = build_data(root, domain, seed, config["adaptation_limit"], config["max_history"])
    if not config["diagnostic"] and data.audit.get("synthetic") is not False:
        raise ValueError("Formal Qwen runs require real data; fixtures must use diagnostic=True")
    base_text, images, base_metadata = _base_features(root, domain, data, model_path, device, config)
    # Runtime/checkpoint frequency do not change mathematical training state.
    fingerprint_config = {key: value for key, value in config.items() if key not in {"max_runtime_seconds", "checkpoint_steps", "export_batch_size"}}
    fingerprint = _json_hash({"protocol_version": 1, "domain": domain, "seed": seed, "config": fingerprint_config,
                              "data_hashes": data.hashes, "base_features_sha256": base_metadata["features_sha256"], "model_id": MODEL_ID, "revision": REVISION})
    shared = root / "artifacts/qwen" / domain / "warmup" / f"seed{seed}"
    orders, negatives, sampling_hash = _sampling(shared, data, seed, config, fingerprint)
    # HF model construction during a cold base-cache fill may consume Torch RNG.
    # Warmup initialization must be identical to an already-cached invocation.
    seed_everything(seed)
    warmup_model, warmup = _warmup(root, domain, seed, data, base_text, images, device, config, fingerprint, started)
    anchors = None
    results = []
    for mode in MODES:
        if _deadline_reached(started, config):
            raise ResumableStop("Previous stages saved; next mode will begin in a fresh allocated window")
        if mode == "qwen_semantic":
            anchors = _review_anchors(shared, data, model_path, device, config, fingerprint)
        results.append(_train_mode(root, domain, seed, mode, data, base_text, images, base_metadata, warmup_model, warmup,
                                  orders, negatives, anchors, model_path, device, config, fingerprint, started, sampling_hash))
        anchors = None
    return {"domain": domain, "seed": seed, "status": "diagnostic_complete" if config["diagnostic"] else "complete", "units": results}
