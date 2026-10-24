"""Local Shopping Studio encoders over verified, frozen base feature caches.

Models load only when encode_query/encode_image is explicitly called. No network
requests, adapters, training, result writes or synthetic fallbacks occur here.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import threading
import warnings
from contextlib import contextmanager, nullcontext
from functools import lru_cache
from pathlib import Path

import numpy as np

from .qwen_text import MODEL_ID as QWEN_ID, REVISION as QWEN_REVISION

CLIP_ID = "openai/clip-vit-base-patch32"
CLIP_REVISION = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
QUERY_INSTRUCTION = "Given a shopping request, retrieve product descriptions that match the user's needs."
MAX_LENGTH = 256
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000
MAX_IMAGE_SIDE = 8192
DOMAINS = ("baby", "sports", "clothing")
_ENCODE_LOCK = threading.RLock()


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_signature(paths) -> tuple:
    return tuple((str(path), path.stat().st_size, path.stat().st_mtime_ns) for path in paths)


def _root(root) -> Path:
    return Path(root).expanduser().resolve()


def _device() -> str:
    device = os.environ.get("SHOPLENS_STUDIO_DEVICE", "cpu").lower()
    if device not in ("cpu", "mps"):
        raise ValueError("SHOPLENS_STUDIO_DEVICE must be cpu or mps.")
    return device


def _snapshot(root: Path, model_id: str, revision: str, *, project_only: bool) -> Path | None:
    caches = [root / "data/studio_hf_cache"]
    if not project_only:
        if os.environ.get("HF_HUB_CACHE"):
            caches.append(Path(os.environ["HF_HUB_CACHE"]).expanduser())
        caches.append(Path.home() / ".cache/huggingface/hub")
    repository = "models--" + model_id.replace("/", "--")
    for cache in caches:
        path = cache / repository / "snapshots" / revision
        weights = list(path.glob("*.safetensors")) + list(path.glob("pytorch_model*.bin"))
        required = ("config.json", "tokenizer.json", "tokenizer_config.json")
        if model_id == CLIP_ID:
            required += ("preprocessor_config.json",)
        if weights and all(file.is_file() for file in weights) and all((path / name).is_file() for name in required):
            return path.resolve()
    return None


def status(root) -> dict:
    """Inspect local preparation only; never imports or loads a model."""
    root = _root(root)
    qwen = _snapshot(root, QWEN_ID, QWEN_REVISION, project_only=True)
    clip = _snapshot(root, CLIP_ID, CLIP_REVISION, project_only=False)
    message = "Run python scripts/prepare_studio_models.py --root <project-root> before searching."
    return {
        "ready": bool(qwen and clip), "query_ready": bool(qwen), "image_ready": bool(clip), "device": _device(),
        "qwen": {"ready": bool(qwen), "path": str(qwen) if qwen else None,
                 "model_id": QWEN_ID, "revision": QWEN_REVISION, "error": None if qwen else f"Pinned Qwen model is not prepared locally. {message}"},
        "clip": {"ready": bool(clip), "path": str(clip) if clip else None,
                 "model_id": CLIP_ID, "revision": CLIP_REVISION, "error": None if clip else f"Pinned CLIP model is not prepared locally. {message} Add --include-clip if needed."},
        "query_instruction": QUERY_INSTRUCTION, "max_length": MAX_LENGTH,
        "padding_side": "left", "pooling": "last valid token", "normalization": "L2",
        "preparation_check": "Local files present; models are validated when explicitly loaded.",
    }


def load_catalog(root, domain: str) -> dict:
    """Return read-only base towers after hashes, provenance and order checks."""
    if domain not in DOMAINS:
        raise ValueError("Unknown Shopping Studio domain.")
    root = _root(root)
    paths = [root / "data/processed" / domain / "items.jsonl",
             root / "artifacts/qwen" / domain / "base/features.npz",
             root / "artifacts/qwen" / domain / "base/features.json",
             root / "artifacts" / domain / "features.npz",
             root / "artifacts" / domain / "features.json"]
    if any(not path.is_file() for path in paths):
        raise FileNotFoundError("Shopping Studio requires the existing catalog, Qwen base cache and original CLIP provenance.")
    return _catalog_cached(str(root), domain, _file_signature(paths))


@lru_cache(maxsize=3)
def _catalog_cached(root: str, domain: str, signature: tuple) -> dict:
    paths = [Path(entry[0]) for entry in signature]
    catalog_path, base_path, base_meta_path, old_path, old_meta_path = paths
    base_meta = json.loads(base_meta_path.read_text(encoding="utf-8"))
    old_meta = json.loads(old_meta_path.read_text(encoding="utf-8"))
    identity = base_meta.get("identity", {})
    catalog_sha, base_sha, old_sha = _sha(catalog_path), _sha(base_path), _sha(old_path)
    if identity.get("items_sha256") != catalog_sha or old_meta.get("catalog_sha256") != catalog_sha:
        raise ValueError("Catalog SHA256 does not match the recorded feature provenance.")
    if base_meta.get("features_sha256") != base_sha:
        raise ValueError("Qwen base feature cache SHA256 mismatch.")
    if identity.get("old_features_sha256") != old_sha or old_meta.get("features_sha256") != old_sha:
        raise ValueError("Original CLIP feature cache SHA256 mismatch.")
    for metadata in (base_meta, identity):
        if metadata.get("model_id") != QWEN_ID or metadata.get("revision") != QWEN_REVISION:
            raise ValueError("Qwen base feature model identity does not match the pinned encoder.")
    expected_encoding = {"max_length": 256, "padding_side": "left", "pooling": "last valid token", "normalization": "L2", "truncation": True}
    if base_meta.get("encoding_config") != expected_encoding:
        raise ValueError("Qwen base feature encoding protocol mismatch.")
    if old_meta.get("clip_model") != CLIP_ID or old_meta.get("image_preprocessing") != "CLIPProcessor + EXIF transpose":
        raise ValueError("Original CLIP model or image preprocessing does not match Studio.")
    if base_meta.get("synthetic") or base_meta.get("diagnostic") or identity.get("diagnostic") or old_meta.get("synthetic"):
        raise ValueError("Shopping Studio requires real, non-diagnostic base feature caches.")
    items = [json.loads(line) for line in catalog_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    item_ids = [str(item["item_id"]) for item in items]
    if not item_ids or len(set(item_ids)) != len(item_ids):
        raise ValueError("Catalog must contain distinct product IDs.")
    with np.load(base_path, allow_pickle=False) as archive:
        if archive["item_ids"].astype(str).tolist() != item_ids:
            raise ValueError("Qwen base feature item order does not match the catalog.")
        text = np.asarray(archive["text"], dtype=np.float32)
        image = np.asarray(archive["image"], dtype=np.float32)
    with np.load(old_path, allow_pickle=False) as archive:
        if archive["item_ids"].astype(str).tolist() != item_ids or not np.array_equal(archive["image"], image):
            raise ValueError("Base image features do not match the original aligned CLIP cache.")
        if "image_present" in archive and not np.all(archive["image_present"]):
            raise ValueError("The catalog contains missing image features.")
    for name, matrix, width in (("text", text, 1024), ("image", image, 512)):
        if matrix.shape != (len(items), width) or not np.isfinite(matrix).all():
            raise ValueError(f"Invalid {name} base feature dimensions or values.")
        if not np.allclose(np.linalg.norm(matrix, axis=1), 1, atol=1e-3, rtol=1e-3):
            raise ValueError(f"Base {name} features must already be L2-normalized.")
        matrix.flags.writeable = False
    return {"items": items, "item_ids": item_ids, "text": text, "image": image,
            "signature": (domain, catalog_sha, base_sha, old_sha),
            "metadata": {"domain": domain, "model_id": QWEN_ID, "revision": QWEN_REVISION,
                         "clip_model": CLIP_ID, "clip_local_revision": CLIP_REVISION,
                         "clip_revision_note": "Original features record the CLIP model ID and preprocessing, but not a CLIP revision.",
                         "encoding_config": expected_encoding, "catalog_sha256": catalog_sha, "base_features_sha256": base_sha,
                         "query_instruction": QUERY_INSTRUCTION, "ranking": "Frozen base embedding cosine; not a formal user-preference test."}}


def _model_path(root, kind: str) -> tuple[str, str, tuple]:
    details = status(root)[kind]
    if not details["ready"]:
        raise RuntimeError(details["error"])
    path = Path(details["path"])
    files = sorted(file for file in path.iterdir() if file.is_file() and file.suffix in (".json", ".safetensors", ".bin", ".txt"))
    return str(path), _device(), _file_signature(files)


@lru_cache(maxsize=1)
def _query_model(path: str, device: str, signature: tuple):
    import torch
    from transformers import AutoModel, AutoTokenizer

    del signature
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable. Set SHOPLENS_STUDIO_DEVICE=cpu.")
    tokenizer = AutoTokenizer.from_pretrained(path, padding_side="left", local_files_only=True, token=False, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModel.from_pretrained(path, torch_dtype=torch.float32, attn_implementation="sdpa", local_files_only=True, token=False, trust_remote_code=False).to(device).eval()
    model.requires_grad_(False)
    model.config.use_cache = False
    if getattr(model.config, "hidden_size", None) != 1024:
        raise ValueError("Prepared Qwen model does not provide 1024-dimensional embeddings.")
    return model, tokenizer


@lru_cache(maxsize=1)
def _query_runtime(path: str, device: str, signature: tuple) -> dict:
    model, tokenizer = _query_model(path, device, signature)
    return {"model": model, "tokenizer": tokenizer, "peft": False, "adapters": {}}


@contextmanager
def _query_context(runtime: dict, adapter_name: str | None = None):
    """Called inside _ENCODE_LOCK; leave the shared base disabled on every exit."""
    model = runtime["model"]
    if not runtime["peft"]:
        yield model
        return
    previous = model.active_adapter
    try:
        if adapter_name is not None:
            model.set_adapter(adapter_name)
            model.base_model.enable_adapter_layers()
        model.eval().requires_grad_(False)
        with model.disable_adapter() if adapter_name is None else nullcontext():
            yield model
    finally:
        try:
            model.set_adapter(previous)
        finally:
            model.base_model.disable_adapter_layers()
            model.eval().requires_grad_(False)


def _ensure_adapter(runtime: dict, name: str, adapter_path: str):
    if name in runtime["adapters"]:
        return
    from peft import PeftModel

    try:
        if runtime["peft"]:
            runtime["model"].load_adapter(adapter_path, adapter_name=name, is_trainable=False, local_files_only=True)
        else:
            runtime["model"] = PeftModel.from_pretrained(runtime["model"], adapter_path,
                                                       adapter_name=name, is_trainable=False, local_files_only=True)
            runtime["peft"] = True
        runtime["model"].eval().requires_grad_(False)
        runtime["model"].base_model.disable_adapter_layers()
        runtime["adapters"][name] = adapter_path
    except Exception:
        # A failed PEFT load may already have injected layers. Quarantine that
        # shared instance; do not return a base-vector substitute for this mode.
        from peft.tuners.tuners_utils import BaseTunerLayer
        for layer in runtime["model"].modules():
            if isinstance(layer, BaseTunerLayer):
                layer.enable_adapters(False)
        _query_runtime.cache_clear()
        _query_model.cache_clear()
        raise


def _normalized(vector, width: int) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if vector.shape != (width,) or not np.isfinite(vector).all() or not np.isfinite(norm) or norm <= 0:
        raise ValueError("Encoder returned an invalid embedding; no fallback is used.")
    vector = vector / norm
    vector.flags.writeable = False
    return vector


def _query_text(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Enter a nonempty shopping request.")
    text = text.strip()
    if len(text) > 8000:
        raise ValueError("Shopping requests must be at most 8,000 characters; encoding truncates to 256 tokens.")
    return text


def encode_query(root, text: str) -> np.ndarray:
    """Encode one shopping request with frozen Qwen, including after LoRA use."""
    text = _query_text(text)
    path, device, signature = _model_path(root, "qwen")
    with _ENCODE_LOCK:
        return _query_cached(path, device, signature, text, QUERY_INSTRUCTION).copy()


@lru_cache(maxsize=32)
def _query_cached(path: str, device: str, signature: tuple, text: str, instruction: str) -> np.ndarray:
    import torch
    from .qwen_text import embed_texts, review_query

    runtime = _query_runtime(path, device, signature)
    with _query_context(runtime), torch.inference_mode():
        vector = embed_texts(runtime["model"], runtime["tokenizer"], [review_query(text, instruction)], device, max_length=MAX_LENGTH)[0].cpu().numpy()
    return _normalized(vector, 1024)


def _export_api():
    # studio_models itself consumes load_catalog, so keep this import lazy.
    from . import studio_models
    return studio_models


def query_encoding_metadata(root, domain: str) -> dict:
    """Verified query/candidate identities; this does not load a language model."""
    api = _export_api()
    protocol = {"instruction": QUERY_INSTRUCTION, "max_length": MAX_LENGTH, "padding_side": "left",
                "pooling": "last valid token", "normalization": "L2", "truncation": True, "compute_dtype": "float32"}
    protocol_hash = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    result = {}
    for mode in api.MODES:
        exported = api.export_metadata(root, domain, mode)
        if exported.get("model_id") != QWEN_ID or exported.get("revision") != QWEN_REVISION:
            raise ValueError("Selected query/candidate export does not match the pinned local Qwen base.")
        identity = {"domain": domain, "model": mode, "model_id": QWEN_ID, "revision": QWEN_REVISION,
                    "adapter_sha256": exported["adapter_sha256"], "artifact_signature": exported["artifact_signature"],
                    "query_protocol_hash": protocol_hash}
        result[mode] = {**exported, "instruction": QUERY_INSTRUCTION, "encoding_protocol": dict(protocol),
                        "query_protocol_hash": protocol_hash,
                        "encoder_signature": hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest(),
                        "space": "Frozen Qwen base" if mode == "qwen_frozen" else "Same selected candidate LoRA encoder used for the query",
                        "scope": "Exploratory shopping-request retrieval; not the formal history-to-candidate evaluation protocol."}
    return result


def encode_queries(root, domain: str, text: str) -> dict:
    """Return all three matched-encoder query vectors, or raise without fallback."""
    text = _query_text(text)
    metadata = query_encoding_metadata(root, domain)
    api = _export_api()
    adapter_paths = {mode: api.adapter_path(root, domain, mode) for mode in api.MODES}
    for mode in api.MODES:
        if mode != "qwen_frozen" and adapter_paths[mode] is None:
            raise RuntimeError(f"The selected {mode} adapter is unavailable; no fallback is used.")
    path, device, signature = _model_path(root, "qwen")
    with _ENCODE_LOCK:
        result = {"qwen_frozen": _query_cached(path, device, signature, text, QUERY_INSTRUCTION).copy()}
        for mode in api.MODES:
            if mode == "qwen_frozen":
                continue
            details = metadata[mode]
            result[mode] = _adapted_query_cached(path, device, signature, domain, mode,
                                                details["adapter_sha256"], details["artifact_signature"],
                                                details["query_protocol_hash"], str(adapter_paths[mode]),
                                                QUERY_INSTRUCTION, text).copy()
        return result


@lru_cache(maxsize=64)
def _adapted_query_cached(path: str, device: str, signature: tuple, domain: str, mode: str,
                          adapter_sha: str, artifact_signature: str, protocol_hash: str,
                          adapter_path: str, instruction: str, text: str) -> np.ndarray:
    import torch
    from .qwen_text import embed_texts, review_query

    runtime = _query_runtime(path, device, signature)
    name = "studio_" + hashlib.sha256(f"{domain}|{mode}|{adapter_sha}|{artifact_signature}".encode()).hexdigest()[:24]
    _ensure_adapter(runtime, name, adapter_path)
    with _query_context(runtime, name), torch.inference_mode():
        vector = embed_texts(runtime["model"], runtime["tokenizer"], [review_query(text, instruction)], device, max_length=MAX_LENGTH)[0].cpu().numpy()
    return _normalized(vector, 1024)


def _decoded_image(image_bytes: bytes):
    from PIL import Image, ImageOps, UnidentifiedImageError

    if not isinstance(image_bytes, bytes) or not image_bytes:
        raise ValueError("Upload a nonempty PNG, JPEG or WebP image.")
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise ValueError("Reference images must be at most 10 MB.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(image_bytes)) as picture:
                if picture.format not in ("PNG", "JPEG", "WEBP") or getattr(picture, "n_frames", 1) != 1:
                    raise ValueError("Use a single-frame PNG, JPEG or WebP image.")
                if max(picture.size) > MAX_IMAGE_SIDE or picture.width * picture.height > MAX_IMAGE_PIXELS:
                    raise ValueError("Reference image exceeds 20 megapixels or an 8,192-pixel side.")
                picture.verify()
            with Image.open(io.BytesIO(image_bytes)) as picture:
                return ImageOps.exif_transpose(picture).convert("RGB").copy()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ValueError("The uploaded image is invalid or exceeds safe decoding limits.") from exc


@lru_cache(maxsize=1)
def _image_model(path: str, device: str, signature: tuple):
    import torch
    from transformers import CLIPModel, CLIPProcessor

    del signature
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable. Set SHOPLENS_STUDIO_DEVICE=cpu.")
    processor = CLIPProcessor.from_pretrained(path, use_fast=False, local_files_only=True, token=False)
    model = CLIPModel.from_pretrained(path, local_files_only=True, token=False).to(device).eval()
    model.requires_grad_(False)
    return model, processor


def encode_image(root, image_bytes: bytes) -> np.ndarray:
    """Encode image bytes in memory with the catalog's CLIP preprocessing."""
    picture = _decoded_image(image_bytes)
    picture.close()
    path, device, signature = _model_path(root, "clip")
    with _ENCODE_LOCK:
        return _image_cached(path, device, signature, image_bytes).copy()


@lru_cache(maxsize=8)
def _image_cached(path: str, device: str, signature: tuple, image_bytes: bytes) -> np.ndarray:
    import torch

    model, processor = _image_model(path, device, signature)
    with _decoded_image(image_bytes) as picture:
        inputs = processor(images=[picture], return_tensors="pt").to(device)
    with torch.inference_mode():
        vector = model.get_image_features(**inputs)
        if not isinstance(vector, torch.Tensor):
            vector = vector.pooler_output
    return _normalized(vector[0].float().cpu().numpy(), 512)
