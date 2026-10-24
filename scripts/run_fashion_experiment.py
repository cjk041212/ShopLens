#!/usr/bin/env python3
"""Fetch a fixed FashionIQ subset, freeze CLIP, train/evaluate a fusion head.

The bounded protocol is fixed before image/feature access. Official validation
is held out from all optimizer/checkpoint choices. No manual annotation occurs.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import time
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from shoplens.fashion_retrieval import (ANNOTATION_REVISION, CATEGORIES, CLIP_REVISION, IMAGE_URL_REVISION,
    IMAGE_INTEGRITY_POLICY, METHODS, build_fusion, enforce_image_hash_disjoint, image_ids, normalized,
    rank_targets, retrieval_metrics, select_pairs)

PROTOCOL = {"seed": 42, "train_per_category": 300, "dev_per_category": 30, "evaluation_per_category": 100,
            "distractors_per_category": 100, "epochs": 40, "batch_size": 64,
            "learning_rate": 0.001, "weight_decay": 0.01, "temperature": 0.07,
            "hidden_dimension": 128, "checkpoint_selection": "internal-dev macro recall@10; earliest tied epoch",
            "image_integrity_policy": IMAGE_INTEGRITY_POLICY,
            "evaluation_pool": "category-specific union of selected reference/targets plus 100 fixed distractors",
            "scope": "bounded available-image FashionIQ subset; not the official full-gallery leaderboard"}


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def fetch(url, *, limit=20_000_000):
    request = urllib.request.Request(url, headers={"User-Agent": "ShopLens-FashionIQ-coursework/1.0"})
    with urllib.request.urlopen(request, timeout=20) as response:
        data = response.read(limit + 1)
    if len(data) > limit:
        raise ValueError("Download exceeds the bounded size limit.")
    return data


def prepare(root):
    from PIL import Image
    directory = root / "assets/benchmarks/fashion"
    directory.mkdir(parents=True, exist_ok=True)
    provenance = []
    annotations, splits, urls = {}, {}, {}
    for category in CATEGORIES:
        for split in ("train", "val"):
            for kind, stem in (("captions", "cap"), ("image_splits", "split")):
                url = f"https://raw.githubusercontent.com/XiaoxiaoGuo/fashion-iq/{ANNOTATION_REVISION}/{kind}/{stem}.{category}.{split}.json"
                data = fetch(url)
                provenance.append({"url": url, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)})
                (annotations if kind == "captions" else splits)[category, split] = json.loads(data)
        url = f"https://raw.githubusercontent.com/hongwang600/fashion-iq-metadata/{IMAGE_URL_REVISION}/image_url/asin2url.{category}.txt"
        data = fetch(url)
        provenance.append({"url": url, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)})
        urls.update(dict(line.split() for line in data.decode().splitlines() if line.strip()))
    selected = {"train": [], "dev": [], "evaluation": []}
    used_train, used_dev = set(), set()
    pools = {"train": {}, "dev": {}, "evaluation": {}}
    for category in CATEGORIES:
        selected["train"].extend(select_pairs(annotations[category, "train"], category, "train", PROTOCOL["train_per_category"]))
    used_train = image_ids(selected["train"])
    for category in CATEGORIES:
        selected["dev"].extend(select_pairs(annotations[category, "train"], category, "dev", PROTOCOL["dev_per_category"], forbidden_ids=used_train))
    used_dev = image_ids(selected["dev"])
    for category in CATEGORIES:
        selected["evaluation"].extend(select_pairs(annotations[category, "val"], category, "val", PROTOCOL["evaluation_per_category"], forbidden_ids=used_train | used_dev))
    for phase, rows in selected.items():
        for index, row in enumerate(rows):
            row["query_id"] = f"{phase}-{row['category']}-{index:04d}"
        for category in CATEGORIES:
            ids = image_ids([row for row in rows if row["category"] == category])
            if phase != "train":
                candidates = splits[category, "val" if phase == "evaluation" else "train"]
                forbidden = used_train | used_dev if phase == "evaluation" else used_train
                extra = sorted((item for item in candidates if item not in ids and item not in forbidden),
                               key=lambda item: hashlib.sha256(f"42|{phase}|{category}|{item}".encode()).hexdigest())
                ids.update(extra[:PROTOCOL["distractors_per_category"]])
            pools[phase][category] = sorted(ids)
    all_ids = sorted({item for pool in pools.values() for ids in pool.values() for item in ids})
    missing_urls = [item for item in all_ids if item not in urls]
    print(json.dumps({"phase": "selected-before-download", "images": len(all_ids), "queries": {key: len(value) for key, value in selected.items()}, "missing_urls": len(missing_urls)}), flush=True)
    images = directory / "images"
    images.mkdir(exist_ok=True)
    image_records, failures = {}, []
    def one(item):
        if item not in urls:
            raise ValueError("No URL in official-linked image metadata")
        url = urls[item].replace("http://", "https://", 1)
        path = images / f"{item}.jpg"
        data = path.read_bytes() if path.is_file() else fetch(url, limit=3_000_000)
        with Image.open(io.BytesIO(data)) as image:
            image.verify()
        if not path.is_file():
            path.write_bytes(data)
        return {"path": path.relative_to(root).as_posix(), "url": url,
                "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=12) as executor:
        futures = {executor.submit(one, item): item for item in all_ids}
        for future in as_completed(futures):
            item = futures[future]
            try:
                image_records[item] = future.result()
            except Exception as error:
                failures.append({"image_id": item, "error": f"{type(error).__name__}: {error}"})
            done = len(image_records) + len(failures)
            if done % 300 == 0:
                print(json.dumps({"phase": "download", "done": done, "successful": len(image_records), "failed": len(failures), "seconds": round(time.perf_counter()-started, 2)}), flush=True)
    for phase in selected:
        selected[phase] = [row for row in selected[phase] if row["reference"] in image_records and row["target"] in image_records]
        for category in CATEGORIES:
            pools[phase][category] = [item for item in pools[phase][category] if item in image_records]
        if any(len([row for row in selected[phase] if row["category"] == category]) < 10 for category in CATEGORIES):
            raise ValueError(f"Too few accessible images in {phase}; stopping instead of fabricating a benchmark.")
    assert not image_ids(selected["train"]) & image_ids(selected["dev"])
    assert not (image_ids(selected["train"]) | image_ids(selected["dev"])) & image_ids(selected["evaluation"])
    manifest = {"schema_version": 1, "dataset": "FashionIQ", "source": "https://github.com/XiaoxiaoGuo/fashion-iq",
                "annotation_revision": ANNOTATION_REVISION, "image_url_revision": IMAGE_URL_REVISION,
                "terms": "Official repository states Community Data License Agreement. Original image rights and upstream terms remain applicable.",
                "protocol": PROTOCOL, "sources": provenance, "queries": selected, "candidate_pools": pools,
                "images": dict(sorted(image_records.items())), "download_failures": sorted(failures, key=lambda row: row["image_id"]),
                "selection": "Hash-ordered fixed pairs chosen before image access; unavailable image pairs omitted without replacement.",
                "split_overlap_images": 0}
    manifest = enforce_image_hash_disjoint(manifest)
    write_json(directory / "manifest.json", manifest)
    return manifest


def encode(root, manifest, clip_path, *, threads=2, batch_size=16):
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor
    torch.set_num_threads(threads)
    model = CLIPModel.from_pretrained(clip_path, local_files_only=True).eval()
    processor = CLIPProcessor.from_pretrained(clip_path, local_files_only=True, use_fast=False)
    ids = sorted(manifest["images"])
    all_queries = [row for phase in ("train", "dev", "evaluation") for row in manifest["queries"][phase]]
    image_features, text_features = [], []
    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(ids), batch_size):
            opened = [Image.open(root / manifest["images"][item]["path"]).convert("RGB") for item in ids[start:start+batch_size]]
            inputs = processor(images=opened, return_tensors="pt")
            features = model.get_image_features(**inputs)
            image_features.append(torch.nn.functional.normalize(features, dim=-1).cpu().numpy())
            for image in opened:
                image.close()
            if start % (batch_size * 10) == 0:
                print(json.dumps({"phase": "image-encoding", "done": min(start+batch_size, len(ids)), "total": len(ids), "seconds": round(time.perf_counter()-started, 2)}), flush=True)
        for start in range(0, len(all_queries), 64):
            inputs = processor(text=[row["text"] for row in all_queries[start:start+64]], padding=True,
                               truncation=True, return_tensors="pt")
            features = model.get_text_features(**inputs)
            text_features.append(torch.nn.functional.normalize(features, dim=-1).cpu().numpy())
    output = root / "assets/benchmarks/fashion/features.npz"
    np.savez_compressed(output, image_ids=np.asarray(ids), image_features=np.concatenate(image_features),
                        query_ids=np.asarray([row["query_id"] for row in all_queries]), text_features=np.concatenate(text_features))
    write_json(output.with_suffix(".json"), {"model_id": "openai/clip-vit-base-patch32", "revision": CLIP_REVISION,
            "frozen": True, "dimension": 512, "features_sha256": sha_file(output), "images": len(ids),
            "queries": len(all_queries), "device": "cpu", "torch_threads": threads,
            "encoding_seconds": time.perf_counter()-started})


def experiment(root, manifest):
    if not manifest.get("data_integrity_amendment"):
        raise ValueError("Prepare the final FashionIQ manifest with deterministic image-byte separation before training.")
    manifest = enforce_image_hash_disjoint(manifest)
    import torch
    torch.set_num_threads(4)
    torch.manual_seed(PROTOCOL["seed"])
    np.random.seed(PROTOCOL["seed"])
    torch.use_deterministic_algorithms(True)
    features_path = root / "assets/benchmarks/fashion/features.npz"
    meta = json.loads(features_path.with_suffix(".json").read_text())
    if meta["features_sha256"] != sha_file(features_path):
        raise ValueError("Feature cache checksum mismatch")
    with np.load(features_path, allow_pickle=False) as data:
        image_features = data["image_features"].copy()
        text_features = data["text_features"].copy()
        image_index = {item: index for index, item in enumerate(data["image_ids"].tolist())}
        text_index = {item: index for index, item in enumerate(data["query_ids"].tolist())}
    rows = manifest["queries"]
    tensors = {}
    for phase in rows:
        tensors[phase] = (torch.from_numpy(np.stack([image_features[image_index[row["reference"]]] for row in rows[phase]])),
                          torch.from_numpy(np.stack([text_features[text_index[row["query_id"]]] for row in rows[phase]])))
    model = build_fusion(hidden=PROTOCOL["hidden_dimension"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=PROTOCOL["learning_rate"], weight_decay=PROTOCOL["weight_decay"])
    train_ids = sorted({item for ids in manifest["candidate_pools"]["train"].values() for item in ids})
    train_image_matrix = torch.from_numpy(np.stack([image_features[image_index[item]] for item in train_ids]))
    train_index = {item: index for index, item in enumerate(train_ids)}
    category_masks = {category: torch.tensor([item in set(manifest["candidate_pools"]["train"][category]) for item in train_ids]) for category in CATEGORIES}
    curve, best, best_state = [], -1.0, None
    generator = torch.Generator().manual_seed(PROTOCOL["seed"])

    def evaluate(phase, method):
        ref, txt = tensors[phase]
        if method == "image_only":
            encoded = ref.numpy()
        elif method == "text_only":
            encoded = txt.numpy()
        elif method == "fixed_fusion":
            encoded = normalized(ref.numpy() + txt.numpy())
        else:
            model.eval()
            with torch.inference_mode():
                encoded = model(ref, txt).numpy()
        measured = []
        details = []
        for category in CATEGORIES:
            indexes = [index for index, row in enumerate(rows[phase]) if row["category"] == category]
            selected = [rows[phase][index] for index in indexes]
            candidate_ids = manifest["candidate_pools"][phase][category]
            matrix = np.stack([image_features[image_index[item]] for item in candidate_ids])
            ranks, rankings = rank_targets(encoded[indexes], matrix, candidate_ids, selected)
            metrics = retrieval_metrics(ranks)
            measured.append({"category": category, "method": method, "candidate_count": len(candidate_ids), **metrics})
            details.extend({**row, "target_rank": rank, "top_ids": ranked[:12]} for row, rank, ranked in zip(selected, ranks, rankings))
        return measured, details

    started = time.perf_counter()
    for epoch in range(PROTOCOL["epochs"] + 1):
        loss_sum, batches = 0.0, 0
        if epoch:
            model.train()
            order = torch.randperm(len(rows["train"]), generator=generator)
            for start in range(0, len(order), PROTOCOL["batch_size"]):
                indices = order[start:start+PROTOCOL["batch_size"]]
                ref, txt = tensors["train"]
                query = model(ref[indices], txt[indices])
                logits = query @ train_image_matrix.T / PROTOCOL["temperature"]
                targets = []
                for batch_row, index in enumerate(indices.tolist()):
                    row = rows["train"][index]
                    logits[batch_row, ~category_masks[row["category"]]] = -1e9
                    logits[batch_row, train_index[row["reference"]]] = -1e9
                    targets.append(train_index[row["target"]])
                loss = torch.nn.functional.cross_entropy(logits, torch.tensor(targets))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_sum += float(loss.detach())
                batches += 1
        dev, _ = evaluate("dev", "trained_fusion")
        score = float(np.mean([value["recall@10"] for value in dev]))
        curve.append({"epoch": epoch, "train_loss": loss_sum/batches if batches else None,
                      "dev_macro_recall@10": score, "dev_macro_recall@50": float(np.mean([value["recall@50"] for value in dev]))})
        if score > best:
            best, best_epoch = score, epoch
            best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
        if epoch % 5 == 0:
            print(json.dumps({"phase": "train", **curve[-1], "best_epoch": best_epoch}), flush=True)
    model.load_state_dict(best_state)
    model_directory = root / "models/fashion"
    model_directory.mkdir(parents=True, exist_ok=True)
    model_path = model_directory / "fusion.pt"
    torch.save({"state_dict": best_state, "dimension": 512, "hidden_dimension": PROTOCOL["hidden_dimension"],
                "seed": PROTOCOL["seed"], "selected_epoch": best_epoch}, model_path)
    metric_rows, query_results = [], {row["query_id"]: {**row, "methods": {}} for row in rows["evaluation"]}
    for method in METHODS:
        measured, details = evaluate("evaluation", method)
        metric_rows.extend(measured)
        for row in details:
            query_results[row["query_id"]]["methods"][method] = {"target_rank": row["target_rank"], "top_ids": row["top_ids"]}
    aggregate = []
    for method in METHODS:
        records = [row for row in metric_rows if row["method"] == method]
        aggregate.append({"method": method, "queries": sum(row["queries"] for row in records),
                          **{key: float(np.mean([row[key] for row in records])) for key in ("recall@1", "recall@5", "recall@10", "recall@50", "mrr")}})
    output = root / "results/fashion"
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "summary.json", {"schema_version": 1, "status": "complete", "dataset": "FashionIQ",
              "protocol": manifest["protocol"], "selected_epoch": best_epoch, "trainable_parameters": sum(p.numel() for p in model.parameters()),
              "clip_frozen": True, "clip_revision": CLIP_REVISION, "phase_queries": {key: len(value) for key, value in rows.items()},
              "candidate_counts": {phase: {category: len(ids) for category, ids in pools.items()} for phase, pools in manifest["candidate_pools"].items()},
              "image_count": len(manifest["images"]), "download_failures": len(manifest["download_failures"]), "split_overlap_images": 0,
              "split_overlap_image_ids": 0, "split_overlap_image_sha256": 0,
              "data_integrity_amendment": manifest.get("data_integrity_amendment"),
              "macro_metrics": aggregate, "category_metrics": metric_rows, "training_curve": curve,
              "model_path": model_path.relative_to(root).as_posix(), "model_sha256": sha_file(model_path),
              "manifest_sha256": sha_file(root / "assets/benchmarks/fashion/manifest.json"), "features_sha256": sha_file(features_path),
              "training_and_evaluation_seconds": time.perf_counter()-started,
              "limitations": ["Fixed seed42 bounded available-image subset, not a full FashionIQ benchmark or Amazon next-item comparison.",
                "Known FashionIQ target is the single relevance label; other suitable garments can be counted as negatives.",
                "Candidate gallery is category-specific and much smaller than the official full gallery.",
                "CLIP pretraining contamination cannot be ruled out; no claimed causal generalization guarantee.",
                "Official validation is evaluated after train-only development checkpoint selection; no hidden test scores."]})
    write_json(output / "queries.json", list(query_results.values()))
    with (output / "metrics.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(metric_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(metric_rows)
    print(json.dumps({"phase": "complete", "selected_epoch": best_epoch, "macro_metrics": aggregate}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--stage", choices=("all", "download", "encode", "train"), default="all")
    parser.add_argument("--clip-path", type=Path)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    manifest_path = root / "assets/benchmarks/fashion/manifest.json"
    if args.stage in ("all", "download"):
        manifest = prepare(root)
    else:
        manifest = json.loads(manifest_path.read_text())
    if args.stage in ("all", "encode"):
        clip_path = args.clip_path
        if clip_path is None:
            from shoplens.studio_encoders import CLIP_ID, _snapshot
            clip_path = _snapshot(root, CLIP_ID, CLIP_REVISION, project_only=False)
        if clip_path is None:
            raise FileNotFoundError("Prepare the pinned CLIP model first, or pass --clip-path to an existing local snapshot.")
        encode(root, manifest, clip_path, threads=args.threads)
    if args.stage in ("all", "train"):
        experiment(root, manifest)


if __name__ == "__main__":
    main()
