"""Bounded FashionIQ composed-image retrieval with frozen CLIP features.

This experiment uses existing FashionIQ relative captions and target IDs. It is
separate from the Amazon next-item recommendation protocol and needs no new
manual relevance labels. Importing this module never loads models or networks.
"""

from __future__ import annotations

import hashlib
import json
import copy
from functools import lru_cache
from pathlib import Path

import numpy as np

CATEGORIES = ("dress", "shirt", "toptee")
METHODS = ("image_only", "text_only", "fixed_fusion", "trained_fusion")
ANNOTATION_REVISION = "f71b16c3b09ef03b02c0fed160fa072fb3b8aa16"
IMAGE_URL_REVISION = "3bb39d7d9a024d92f1b0fd0929b38d73dff24a0a"
CLIP_REVISION = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
IMAGE_INTEGRITY_POLICY = "Preserve the fixed evaluation gallery, then development gallery; remove cross-phase identical image SHA256s from lower-priority phases and affected query pairs without replacement."


def normalized(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim not in (1, 2) or not np.isfinite(array).all():
        raise ValueError("Expected finite vectors or a matrix.")
    norms = np.linalg.norm(array, axis=-1, keepdims=True)
    if np.any(norms < 1e-12):
        raise ValueError("Cannot normalize a zero vector.")
    return array / norms


def stable_order(rows: list[dict], category: str, split: str, seed: int = 42) -> list[dict]:
    """Choose subsets using only source identities, before computing features."""
    def key(row):
        payload = json.dumps([seed, category, split, row["candidate"], row["target"], row["captions"]], separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()
    return sorted(rows, key=key)


def select_pairs(rows: list[dict], category: str, split: str, count: int, *, forbidden_ids=()) -> list[dict]:
    forbidden = set(forbidden_ids)
    selected = []
    seen = set()
    for row in stable_order(rows, category, split):
        reference, target = row["candidate"], row["target"]
        if reference == target or reference in forbidden or target in forbidden:
            continue
        if (reference, target) in seen or len(row["captions"]) != 2:
            continue
        seen.add((reference, target))
        selected.append({"category": category, "reference": reference, "target": target,
                         "captions": row["captions"], "text": " and ".join(row["captions"])})
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError(f"Insufficient eligible {category}/{split} pairs: {len(selected)} < {count}")
    return selected


def image_ids(rows: list[dict]) -> set[str]:
    return {row[name] for row in rows for name in ("reference", "target")}


def enforce_image_hash_disjoint(manifest: dict) -> dict:
    """Protect evaluation exactly, then development, while removing byte leaks.

    The integrity amendment was introduced after an initial result inspection.
    Its rule depends only on image bytes and phase priority, never labels, ranks
    or model outcomes. Retain that disclosure when reproducing the final run.
    """
    result = copy.deepcopy(manifest)
    pools = result["candidate_pools"]
    images = result["images"]
    phase_ids = lambda phase: set().union(*(set(ids) for ids in pools[phase].values()))
    hashes = lambda ids: {images[item]["sha256"] for item in ids}
    pairs = (("train", "dev"), ("train", "evaluation"), ("dev", "evaluation"))
    initial = {f"{first}_{second}": len(hashes(phase_ids(first)) & hashes(phase_ids(second))) for first, second in pairs}
    if result.get("data_integrity_amendment"):
        if any(initial.values()):
            raise ValueError("An amended FashionIQ manifest still contains cross-phase image byte overlap.")
        return result
    evaluation_queries = copy.deepcopy(result["queries"]["evaluation"])
    evaluation_pools = copy.deepcopy(pools["evaluation"])
    eval_hashes = hashes(phase_ids("evaluation"))
    removed_ids = {"dev": {item for item in phase_ids("dev") if images[item]["sha256"] in eval_hashes}}
    kept_dev = phase_ids("dev") - removed_ids["dev"]
    forbidden_train = eval_hashes | hashes(kept_dev)
    removed_ids["train"] = {item for item in phase_ids("train") if images[item]["sha256"] in forbidden_train}
    removals = {}
    for phase in ("train", "dev"):
        removed = removed_ids[phase]
        excluded = [row for row in result["queries"][phase] if row["reference"] in removed or row["target"] in removed]
        result["queries"][phase] = [row for row in result["queries"][phase] if row["reference"] not in removed and row["target"] not in removed]
        for category in pools[phase]:
            pools[phase][category] = [item for item in pools[phase][category] if item not in removed]
        removals[phase] = {"image_ids": sorted(removed), "query_ids": [row["query_id"] for row in excluded],
                           "image_count": len(removed), "query_count": len(excluded)}
    for first, second in pairs:
        if hashes(phase_ids(first)) & hashes(phase_ids(second)):
            raise ValueError("Cross-phase image bytes remain after the deterministic integrity repair.")
    assert result["queries"]["evaluation"] == evaluation_queries and pools["evaluation"] == evaluation_pools
    amendment = {"reason": "Different FashionIQ product IDs can reuse identical image bytes.",
                 "timing": "Introduced after inspecting the initial experiment results. The deterministic repair is independent of labels, ranks and scores; final results always replace preliminary results, regardless of performance.",
                 "policy": IMAGE_INTEGRITY_POLICY, "priority": ["evaluation", "dev", "train"],
                 "initial_pairwise_hash_overlap_counts": initial,
                 "initial_pairwise_overlap_count_sum": sum(initial.values()), "removed": removals,
                 "evaluation_queries_and_entire_gallery_preserved": True, "replacement_sampling": False}
    result["data_integrity_amendment"] = amendment
    result["protocol"]["image_integrity_policy"] = IMAGE_INTEGRITY_POLICY
    result["protocol"]["data_integrity_amendment"] = amendment
    result["split_overlap_image_ids"] = 0
    result["split_overlap_image_sha256"] = 0
    return result


def rank_targets(query_vectors, candidate_vectors, candidate_ids, queries) -> tuple[list[int], list[list[str]]]:
    """Rank one known target per query, excluding the reference before ranking.

    Ties resolve by candidate ID. This avoids candidate file/download order
    changing a measured rank, and never assigns a fake rank to missing targets.
    """
    ids = list(candidate_ids)
    if len(ids) != len(set(ids)) or len(ids) != len(candidate_vectors):
        raise ValueError("Candidate IDs must be unique and align with vectors.")
    q = normalized(query_vectors)
    images = normalized(candidate_vectors)
    if q.ndim != 2 or images.ndim != 2 or q.shape[0] != len(queries) or q.shape[1] != images.shape[1]:
        raise ValueError("Query and candidate feature shapes do not match.")
    scores = q @ images.T
    id_index = {item: index for index, item in enumerate(ids)}
    ranks, rankings = [], []
    for row, score in zip(queries, scores):
        if row["target"] not in id_index or row["target"] == row["reference"]:
            raise ValueError("A distinct target must be present in the candidate pool.")
        eligible = [index for index, item in enumerate(ids) if item != row["reference"]]
        order = sorted(eligible, key=lambda index: (-float(score[index]), ids[index]))
        ranking = [ids[index] for index in order]
        ranks.append(ranking.index(row["target"]) + 1)
        rankings.append(ranking)
    return ranks, rankings


def retrieval_metrics(ranks) -> dict:
    values = np.asarray(ranks)
    if (values.ndim != 1 or not len(values) or np.any(values < 1)
            or not np.isfinite(values).all() or np.any(values != np.floor(values))):
        raise ValueError("Ranks must be a nonempty vector of positive integers.")
    return {"queries": int(len(values)), "recall@1": float(np.mean(values <= 1)),
            "recall@5": float(np.mean(values <= 5)), "recall@10": float(np.mean(values <= 10)),
            "recall@50": float(np.mean(values <= 50)), "mrr": float(np.mean(1.0 / values)),
            "median_rank": float(np.median(values))}


def fusion_inputs(reference: np.ndarray, text: np.ndarray) -> np.ndarray:
    ref, txt = normalized(reference), normalized(text)
    if ref.ndim != 2 or ref.shape != txt.shape:
        raise ValueError("Fusion requires aligned image and text vectors in the same CLIP space.")
    return np.concatenate((ref, txt, ref * txt), axis=1)


def build_fusion(dimension=512, hidden=128):
    """Residual projection starts exactly at the fixed image+text baseline."""
    import torch
    class ResidualFusion(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.Sequential(torch.nn.Linear(dimension * 3, hidden), torch.nn.GELU(),
                                              torch.nn.Linear(hidden, dimension))
            torch.nn.init.zeros_(self.layers[-1].weight)
            torch.nn.init.zeros_(self.layers[-1].bias)

        def forward(self, reference, text):
            joined = torch.cat((reference, text, reference * text), dim=-1)
            return torch.nn.functional.normalize(reference + text + self.layers(joined), dim=-1)
    return ResidualFusion()


def load_fashion_results(root) -> dict:
    """Read recorded outputs without loading CLIP, downloading or training."""
    root = Path(root).resolve()
    summary_path = root / "results/fashion/summary.json"
    examples_path = root / "results/fashion/queries.json"
    if not summary_path.is_file() or not examples_path.is_file():
        raise FileNotFoundError("Run scripts/run_fashion_experiment.py before opening FashionIQ results.")
    summary = json.loads(summary_path.read_text())
    queries = json.loads(examples_path.read_text())
    if summary.get("status") != "complete" or not queries:
        raise ValueError("FashionIQ results are incomplete.")
    for row in queries:
        if row["reference"] == row["target"] or set(row["methods"]) != set(METHODS):
            raise ValueError("Malformed FashionIQ query result.")
        for value in row["methods"].values():
            if value["target_rank"] < 1 or row["reference"] in value["top_ids"]:
                raise ValueError("Reference leaked into the retrieved candidate list.")
    return {"summary": summary, "queries": queries}


def encode_fashion_text(runtime_root, text: str) -> np.ndarray:
    """Explicit local CLIP text inference; never downloads a model or trains."""
    if not isinstance(text, str) or not text.strip() or len(text) > 2000:
        raise ValueError("Enter a nonempty modification of at most 2,000 characters.")
    from . import studio_encoders
    path, device, signature = studio_encoders._model_path(runtime_root, "clip")
    with studio_encoders._ENCODE_LOCK:
        return _text_cached(path, device, signature, text.strip()).copy()


@lru_cache(maxsize=64)
def _text_cached(path, device, signature, text):
    import torch
    from . import studio_encoders
    model, processor = studio_encoders._image_model(path, device, signature)
    inputs = processor(text=[text], padding=True, truncation=True, max_length=77, return_tensors="pt").to(device)
    with torch.inference_mode():
        value = model.get_text_features(**inputs)
    if not isinstance(value, torch.Tensor):
        value = value.pooler_output
    result = normalized(value[0].float().cpu().numpy())
    if result.shape != (512,):
        raise ValueError("The prepared CLIP text encoder must return 512 dimensions.")
    result.flags.writeable = False
    return result


def _file_sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@lru_cache(maxsize=2)
def _composition_assets(repo_root: str, signature: tuple):
    del signature
    import torch
    root = Path(repo_root)
    manifest = json.loads((root / "assets/benchmarks/fashion/manifest.json").read_text())
    summary = json.loads((root / "results/fashion/summary.json").read_text())
    features_path = root / "assets/benchmarks/fashion/features.npz"
    weights_path = root / "models/fashion/fusion.pt"
    if (summary.get("status") != "complete" or summary["features_sha256"] != _file_sha(features_path)
            or summary["model_sha256"] != _file_sha(weights_path)
            or summary["manifest_sha256"] != _file_sha(root / "assets/benchmarks/fashion/manifest.json")):
        raise ValueError("FashionIQ model, features or candidate manifest checksum mismatch.")
    with np.load(features_path, allow_pickle=False) as data:
        ids, features = data["image_ids"].tolist(), data["image_features"].copy()
    if features.shape != (len(ids), 512) or len(ids) != len(set(ids)):
        raise ValueError("FashionIQ image cache must contain unique aligned 512-dimensional vectors.")
    features = normalized(features)
    index = {item: position for position, item in enumerate(ids)}
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
    if checkpoint["dimension"] != 512 or checkpoint["selected_epoch"] != summary["selected_epoch"]:
        raise ValueError("The trained fusion checkpoint does not match the recorded experiment.")
    model = build_fusion(dimension=512, hidden=checkpoint["hidden_dimension"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval().requires_grad_(False)
    features.flags.writeable = False
    return manifest, features, index, model


def compose_fashion_query(repo_root, runtime_root, reference_id: str, text: str, category: str, *, limit=12) -> dict:
    """Retrieve real images for a newly edited text without inventing labels.

    repo_root supplies the portable features/head; runtime_root supplies an
    already prepared local CLIP snapshot. Reference image and candidate vectors
    come from the same frozen feature cache. Custom requests have no target rank.
    """
    if category not in CATEGORIES or not isinstance(limit, int) or not 1 <= limit <= 50:
        raise ValueError("Choose a FashionIQ category and a result limit from1 to50.")
    if not isinstance(text, str) or not text.strip() or len(text) > 2000:
        raise ValueError("Enter a nonempty modification of at most 2,000 characters.")
    root = Path(repo_root).resolve()
    paths = [root / path for path in ("assets/benchmarks/fashion/manifest.json", "assets/benchmarks/fashion/features.npz",
                                     "models/fashion/fusion.pt", "results/fashion/summary.json")]
    if any(not path.is_file() for path in paths):
        raise FileNotFoundError("The published FashionIQ images/features/fusion checkpoint are required.")
    signature = tuple((path.stat().st_size, path.stat().st_mtime_ns) for path in paths)
    manifest, features, index, model = _composition_assets(str(root), signature)
    pool = manifest["candidate_pools"]["evaluation"][category]
    if reference_id not in pool or reference_id not in index:
        raise ValueError("Choose a reference image from this category's held-out candidate gallery.")
    text_vector = encode_fashion_text(runtime_root, text)
    reference = features[index[reference_id]]
    import torch
    with torch.inference_mode():
        trained = model(torch.from_numpy(reference.copy()[None, :]), torch.from_numpy(text_vector[None, :])).numpy()[0]
    vectors = {"image_only": reference, "text_only": text_vector,
               "fixed_fusion": normalized(reference + text_vector), "trained_fusion": trained}
    ids = [item for item in pool if item != reference_id]
    candidates = np.stack([features[index[item]] for item in ids])
    results = {}
    for method, query in vectors.items():
        scores = candidates @ query
        order = sorted(range(len(ids)), key=lambda position: (-float(scores[position]), ids[position]))[:limit]
        results[method] = {"top_ids": [ids[position] for position in order],
                           "scores": [float(scores[position]) for position in order]}
    return {"reference": reference_id, "text": text.strip(), "category": category, "candidate_count": len(ids),
            "methods": results, "labels_available": False,
            "encoding": {"model_id": "openai/clip-vit-base-patch32", "revision": CLIP_REVISION,
                         "max_length": 77, "truncation": True, "space": "512-dimensional shared frozen CLIP image/text space"},
            "scope": "New custom request: no ground-truth paired target or retrieval metric is available."}
