"""Verified seed42 exports for exploratory request retrieval and history scoring.

Each mode exposes its selected candidate text matrix and compact provenance.
The optional history scorer retains the exported asymmetric towers and user
network; empty history never becomes an invented user vector.
"""

from __future__ import annotations

import hashlib
import json
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from .qwen_inference import _liked_indices, load_scorer
from .qwen_text import MODEL_ID, REVISION
from .studio_encoders import DOMAINS, load_catalog

MODES = ("qwen_frozen", "qwen_semantic", "qwen_preference")
LABELS = dict(zip(MODES, ("Frozen Qwen", "Semantic alignment LoRA", "User preference LoRA")))
SEED = 42


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_hash(directory: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            digest.update(str(path.relative_to(directory)).encode())
            digest.update(_sha(path).encode())
    return digest.hexdigest()


def _paths(root, domain: str, mode: str) -> tuple[Path, list[Path]]:
    if domain not in DOMAINS or mode not in MODES:
        raise ValueError("Unknown Studio domain or trained model.")
    root = Path(root).expanduser().resolve()
    unit = root / "artifacts/qwen" / domain / mode / f"seed{SEED}"
    paths = [root / "data/processed" / domain / "items.jsonl",
             root / "artifacts/qwen" / domain / "base/features.npz",
             root / "artifacts/qwen" / domain / "base/features.json",
             root / "artifacts" / domain / "features.npz",
             root / "artifacts" / domain / "features.json",
             unit / "complete.json", unit / "models" / f"{mode}.json",
             unit / "models" / f"{mode}.pt", unit / "serving.npz",
             unit / "candidate.npz", unit / "results.json"]
    if mode != "qwen_frozen":
        paths.extend((unit / "adapter/adapter_config.json", unit / "adapter/adapter_model.safetensors"))
        paths.extend(path for path in sorted((unit / "adapter").rglob("*")) if path.is_file() and path not in paths)
    if any(not path.is_file() for path in paths):
        raise FileNotFoundError(f"{LABELS[mode]} seed42 is missing a required export or its provenance.")
    return unit, paths


def _signature(root, domain, mode) -> tuple:
    _, paths = _paths(root, domain, mode)
    return tuple((str(path), path.stat().st_size, path.stat().st_mtime_ns, path.stat().st_ctime_ns) for path in paths)


def _signature_id(signature: tuple) -> str:
    return hashlib.sha256(json.dumps(signature, separators=(",", ":")).encode()).hexdigest()


def _integer(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"Invalid trained-model {name}.")
    return value


@lru_cache(maxsize=18)
def _inspect(root: str, domain: str, mode: str, signature: tuple) -> dict:
    unit, _ = _paths(root, domain, mode)
    metadata = json.loads((unit / "models" / f"{mode}.json").read_text(encoding="utf-8"))
    complete = json.loads((unit / "complete.json").read_text(encoding="utf-8"))
    base = json.loads((Path(root) / "artifacts/qwen" / domain / "base/features.json").read_text(encoding="utf-8"))
    if not all(isinstance(record, dict) for record in (metadata, complete, base)):
        raise ValueError("Trained-model metadata must use JSON object records.")
    for record in (metadata, complete):
        if record.get("model") != mode or record.get("seed") != SEED or isinstance(record.get("seed"), bool):
            raise ValueError("Trained-model export model/seed identity mismatch.")
    epoch = _integer(metadata.get("selected_epoch"), "selected epoch")
    if epoch > _integer(metadata.get("epochs"), "epoch budget") or complete.get("selected_epoch") != epoch:
        raise ValueError("Selected epoch does not match the completed export.")
    if metadata.get("family") != "qwen_candidate_lora" or metadata.get("objective") != mode:
        raise ValueError("Trained-model family or objective mismatch.")
    if metadata.get("diagnostic") is not False or not isinstance(metadata.get("data_audit"), dict) or metadata["data_audit"].get("synthetic") is not False:
        raise ValueError("Studio requires completed real-data, non-diagnostic trained exports.")
    if metadata.get("model_id") != MODEL_ID or metadata.get("revision") != REVISION:
        raise ValueError("Trained-model base encoder identity mismatch.")
    if metadata.get("embedding_dim") != 128 or metadata.get("max_history") != 20:
        raise ValueError("Trained-model serving schema must use 128 dimensions and at most 20 history items.")
    if complete.get("fingerprint") != metadata.get("config_sha256") or not metadata.get("config_sha256"):
        raise ValueError("Completed export configuration fingerprint mismatch.")
    for key in ("head_sha256", "serving_sha256", "candidate_sha256"):
        value = metadata.get(key)
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value) or complete.get(key) != value:
            raise ValueError(f"Completed export checksum declaration mismatch: {key}.")
    result_hash = complete.get("results_sha256")
    if not isinstance(result_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", result_hash):
        raise ValueError("Completed export lacks its formal-result checksum declaration.")
    adapter_hash = metadata.get("adapter_sha256")
    if complete.get("adapter_sha256") != adapter_hash:
        raise ValueError("Completed adapter checksum declaration mismatch.")
    if mode == "qwen_frozen":
        if adapter_hash is not None or metadata.get("lora_config") is not None:
            raise ValueError("Frozen export must not declare a trained adapter.")
    elif not isinstance(adapter_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", adapter_hash):
        raise ValueError("Adapted export lacks a valid selected-adapter checksum.")
    if metadata.get("base_features_sha256") != base.get("features_sha256") or metadata.get("features_metadata") != base:
        raise ValueError("Trained export does not match the canonical base feature provenance.")
    return {"metadata": metadata, "complete": complete, "base": base,
            "artifact_signature": _signature_id(signature)}


def model_status(root, domain: str) -> dict:
    """Cheap file/metadata inspection; content checks run before personalization."""
    if domain not in DOMAINS:
        raise ValueError("Unknown Studio domain.")
    root = str(Path(root).expanduser().resolve())
    result = {}
    for mode in MODES:
        details = {"ready": False, "error": None, "seed": SEED, "selected_epoch": None,
                   "objective": mode, "artifact_signature": None,
                   "validation_scope": "Required files and metadata; full content is verified before scoring."}
        try:
            signature = _signature(root, domain, mode)
            details["artifact_signature"] = _signature_id(signature)
            info = _inspect(root, domain, mode, signature)
            details.update(ready=True, selected_epoch=info["metadata"]["selected_epoch"])
        except (OSError, ValueError, KeyError, TypeError) as exc:
            details["error"] = str(exc)
        result[mode] = details
    return result


def _head_schema(path: Path, width: int):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(saved, dict) or not isinstance(saved.get("state_dict"), dict):
        raise ValueError("Trained head must contain its exported state dictionary.")
    state = saved.get("state_dict", {})
    expected = {}
    for side in ("history", "candidate"):
        for modality, source_width in (("text", 1024), ("image", 512)):
            expected[f"{side}_{modality}_projection.weight"] = (width, source_width)
            expected[f"{side}_{modality}_projection.bias"] = (width,)
        expected[f"{side}_fusion.0.weight"] = (width, width * 2)
        for name in ("0.bias", "2.weight", "2.bias"):
            expected[f"{side}_fusion.{name}"] = (width,)
    for layer in (0, 2):
        expected[f"user_network.{layer}.weight"] = (width, width)
        expected[f"user_network.{layer}.bias"] = (width,)
    if set(state) != set(expected) or any(not isinstance(state[key], torch.Tensor) or tuple(state[key].shape) != shape or not torch.isfinite(state[key]).all() for key, shape in expected.items()):
        raise ValueError("Trained head has an invalid tensor schema or nonfinite values.")


@lru_cache(maxsize=9)
def _verified(root: str, domain: str, mode: str, signature: tuple) -> dict:
    info = _inspect(root, domain, mode, signature)
    metadata, complete = info["metadata"], info["complete"]
    unit, _ = _paths(root, domain, mode)
    catalog = load_catalog(root, domain)
    if metadata.get("item_ids") != catalog["item_ids"] or metadata.get("items_sha256") != catalog["metadata"]["catalog_sha256"]:
        raise ValueError("Trained model does not match the canonical catalog or its order.")
    for key, path in (("head_sha256", unit / "models" / f"{mode}.pt"),
                      ("serving_sha256", unit / "serving.npz"), ("candidate_sha256", unit / "candidate.npz")):
        if _sha(path) != metadata[key]:
            raise ValueError(f"Trained artifact content checksum mismatch: {key}.")
    if _sha(unit / "results.json") != complete["results_sha256"]:
        raise ValueError("Completed export formal-result checksum mismatch.")
    if mode != "qwen_frozen" and _tree_hash(unit / "adapter") != metadata["adapter_sha256"]:
        raise ValueError("Selected adapter content checksum mismatch.")
    matrices = []
    with np.load(unit / "candidate.npz", allow_pickle=False) as archive:
        if archive["item_ids"].astype(str).tolist() != catalog["item_ids"]:
            raise ValueError("Selected candidate cache catalog order mismatch.")
        matrices.append(("candidate text", archive["text"].copy(), 1024))
    with np.load(unit / "serving.npz", allow_pickle=False) as archive:
        if archive["item_ids"].astype(str).tolist() != catalog["item_ids"]:
            raise ValueError("Selected serving cache catalog order mismatch.")
        for key in ("history_item_vectors", "candidate_item_vectors"):
            matrices.append((key, archive[key].copy(), 128))
    for name, matrix, width in matrices:
        if matrix.shape != (len(catalog["items"]), width) or not np.issubdtype(matrix.dtype, np.floating) or not np.isfinite(matrix).all() or not np.allclose(np.linalg.norm(matrix, axis=1), 1, atol=1e-3, rtol=1e-3):
            raise ValueError(f"Invalid selected {name} shape, values or normalization.")
    _head_schema(unit / "models" / f"{mode}.pt", 128)
    if _signature(root, domain, mode) != signature:
        raise RuntimeError("Trained artifacts changed while being verified; retry after files are stable.")
    return info


def personalization(root, domain: str, mode: str, positive_ids) -> dict:
    """Return selected trained-head scores in canonical full-catalog order."""
    if isinstance(positive_ids, (str, bytes)):
        raise ValueError("Pass positive product IDs as a sequence of catalog IDs.")
    root = str(Path(root).expanduser().resolve())
    signature = _signature(root, domain, mode)
    info = _verified(root, domain, mode, signature)
    unique_ids = list(dict.fromkeys(str(item_id) for item_id in positive_ids))
    metadata = info["metadata"]
    summary = {"model": mode, "label": LABELS[mode], "seed": SEED,
               "selected_epoch": metadata["selected_epoch"], "objective": metadata["objective"],
               "active": bool(unique_ids), "positive_count": len(unique_ids),
               "history_count": min(len(unique_ids), metadata["max_history"]), "max_history": metadata["max_history"],
               "history_item_ids": list(reversed(unique_ids))[:metadata["max_history"]],
               "history_truncated": len(unique_ids) > metadata["max_history"],
               "artifact_signature": info["artifact_signature"],
               **{key: metadata[key] for key in ("head_sha256", "serving_sha256", "candidate_sha256", "adapter_sha256")},
               "retrieval_note": "History scoring uses the selected export’s base-history tower, selected candidate tower and user head; request retrieval has separate encoding metadata.",
               "score_note": "Selected seed42 trained user-head scores for actual positive catalog history; Studio blends their full-catalog percentiles. These are session scores, not new formal evaluation results." if unique_ids else "Inactive: select at least one More product to use the trained user head. No empty-history scores are invented."}
    if not unique_ids:
        return {"scores": None, "metadata": summary}
    unit, _ = _paths(root, domain, mode)
    scorer = load_scorer(Path(root) / "data/processed" / domain, unit, mode)
    indices = _liked_indices(scorer, unique_ids)
    scores = np.asarray(scorer.score_histories([indices])[0], dtype=np.float32)
    if scores.shape != (len(metadata["item_ids"]),) or not np.isfinite(scores).all():
        raise ValueError("Trained user-head scores are not finite full-catalog scores.")
    return {"scores": scores.copy(), "metadata": summary}


def _export_summary(domain: str, info: dict) -> dict:
    metadata = info["metadata"]
    mode = metadata["model"]
    return {"model": mode, "label": LABELS[mode], "domain": domain, "seed": SEED,
            "selected_epoch": metadata["selected_epoch"], "objective": metadata["objective"],
            "model_id": metadata["model_id"], "revision": metadata["revision"],
            "artifact_signature": info["artifact_signature"],
            "unit": f"artifacts/qwen/{domain}/{mode}/seed{SEED}",
            **{key: metadata[key] for key in ("head_sha256", "serving_sha256", "candidate_sha256", "adapter_sha256")},
            "scope": "Exploratory per-model request retrieval in the selected 1024-dimensional candidate text space; not a formal history-based recommendation benchmark."}


def export_metadata(root, domain: str, mode: str) -> dict:
    """Return verified compact provenance without local absolute paths or arrays."""
    root = str(Path(root).expanduser().resolve())
    signature = _signature(root, domain, mode)
    return _export_summary(domain, _verified(root, domain, mode, signature))


def adapter_path(root, domain: str, mode: str) -> Path | None:
    """Verified local adapter location for internal loading, never export metadata."""
    metadata = export_metadata(root, domain, mode)
    if metadata["adapter_sha256"] is None:
        return None
    return Path(root).expanduser().resolve() / metadata["unit"] / "adapter"


@lru_cache(maxsize=3)
def _candidate_catalog(root: str, domain: str, mode: str, signature: tuple) -> dict:
    info = _verified(root, domain, mode, signature)
    unit, _ = _paths(root, domain, mode)
    with np.load(unit / "candidate.npz", allow_pickle=False) as archive:
        item_ids = archive["item_ids"].astype(str).tolist()
        if item_ids != info["metadata"]["item_ids"]:
            raise ValueError("Selected candidate catalog order changed while loading.")
        selected = np.asarray(archive["text"], dtype=np.float32)
    # Immutable bytes prevent callers from re-enabling writes on a shared cache.
    text = np.frombuffer(selected.tobytes(), dtype=np.float32).reshape(selected.shape)
    if _signature(root, domain, mode) != signature:
        raise RuntimeError("Selected candidate artifacts changed while loading; retry when stable.")
    return {"text": text, "item_ids": tuple(item_ids), "metadata": _export_summary(domain, info)}


def candidate_catalog(root, domain: str, mode: str) -> dict:
    """Return the selected real candidate text matrix in canonical catalog order.

    Image vectors remain in the existing shared base catalog. No missing or
    corrupt selected export falls back to another model or the frozen matrix.
    """
    root = str(Path(root).expanduser().resolve())
    signature = _signature(root, domain, mode)
    cached = _candidate_catalog(root, domain, mode, signature)
    return {"text": cached["text"], "item_ids": list(cached["item_ids"]),
            "metadata": dict(cached["metadata"])}
