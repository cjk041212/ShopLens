#!/usr/bin/env python3
"""Measure the real local Studio engine, with explicit cold/warm boundaries.

The optional public baseline Git ref compares a metadata-only optimization with
identical inputs and requires identical complete recommendation rows. No training
or network calls occur. The first query includes model/adapter loading; later
unique requests and exact cache hits are reported separately.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import resource
import subprocess
import sys
import time
import types
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

QUERIES = [
    "Comfortable cotton shirt for everyday wear",
    "Lightweight jacket for a rainy commute",
    "Black shoes for walking to work",
    "Summer dress with a floral pattern",
    "Warm socks for winter hiking",
    "Blue jeans with a relaxed fit",
    "Small shoulder bag for travel",
    "Soft sweater for a cool evening",
    "Running shorts with pockets",
    "Waterproof boots for wet weather",
    "Simple white t shirt for exercise",
    "Breathable hat for a sunny afternoon",
]


def stats(values):
    import numpy as np
    if not values or any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("Timings must be a nonempty list of finite nonnegative durations.")
    return {"n": len(values), "p50_ms": float(np.median(values)),
            "p95_ms": float(np.percentile(values, 95)), "samples_ms": values}


def catalog_images(root, items, count=12):
    """Use declared catalog paths, supporting both nested and flat image layouts."""
    root, images, identities, seen = Path(root).resolve(), [], [], set()
    for item in items:
        value = item.get("image_path")
        if not isinstance(value, str) or "\\" in value:
            continue
        relative = PurePosixPath(value)
        if relative.is_absolute() or ".." in relative.parts or relative.parts[:2] != ("data", "images"):
            raise ValueError("Catalog image path must remain inside data/images.")
        path = root / value
        if not path.is_file():
            continue
        content = path.read_bytes()
        checksum = hashlib.sha256(content).hexdigest()
        if checksum in seen:
            continue
        seen.add(checksum)
        images.append(content)
        identities.append({"item_id": str(item["item_id"]), "path": value, "sha256": checksum})
        if len(images) == count:
            return images, identities
    raise FileNotFoundError(f"Need {count} distinct catalog images for uncached model forwards.")


def load_baseline(commit, source_root=ROOT):
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("Benchmark baseline must be a complete pinned Git commit SHA.")
    source = subprocess.check_output(["git", "show", "--no-ext-diff", "--no-textconv", f"{commit}:src/shoplens/studio.py"], cwd=source_root).decode()
    baseline = types.ModuleType("shoplens_studio_baseline")
    exec(compile(source, "public_studio_baseline.py", "exec"), baseline.__dict__)
    return baseline, source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "results/performance/studio.json")
    parser.add_argument("--baseline-ref", default="eb53149346207b2500d8e160682e3894fb30125d")
    parser.add_argument("--repeats", type=int, default=12)
    args = parser.parse_args()
    if args.repeats < 10:
        raise ValueError("Use at least ten measurements.")
    thread_environment = {name: "4" for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")}
    os.environ.update(thread_environment)
    import numpy as np
    import torch
    from shoplens import studio, studio_encoders as enc, studio_models as models

    torch.set_num_threads(4)
    os.environ["SHOPLENS_STUDIO_DEVICE"] = "cpu"
    os.environ["HF_HUB_OFFLINE"] = "1"
    root, domain = args.root.resolve(), "clothing"
    records, cache_evidence = [], {}
    baseline, baseline_source = load_baseline(args.baseline_ref)

    def measured(stage, function, repeats=1):
        durations, value = [], None
        for _ in range(repeats):
            started = time.perf_counter()
            value = function()
            durations.append((time.perf_counter() - started) * 1000)
        records.append({"stage": stage, **stats(durations)})
        print(json.dumps({"stage": stage, **{k: v for k, v in stats(durations).items() if k != "samples_ms"}}), flush=True)
        return value

    bundle = measured("cold_catalog_validation", lambda: enc.load_catalog(root, domain))
    candidates = measured("cold_selected_exports", lambda: {mode: models.candidate_catalog(root, domain, mode) for mode in models.MODES})
    measured("warm_catalog_lookup", lambda: enc.load_catalog(root, domain), args.repeats)
    queries = measured("cold_three_query_encoders", lambda: enc.encode_queries(root, domain, QUERIES[0]))
    unique = []
    for query in QUERIES[1:]:
        started = time.perf_counter()
        enc.encode_queries(root, domain, query)
        unique.append((time.perf_counter() - started) * 1000)
    records.append({"stage": "warm_unique_three_queries", **stats(unique)})
    cache_evidence["query_before_repeat"] = {"frozen": enc._query_cached.cache_info()._asdict(), "adapted": enc._adapted_query_cached.cache_info()._asdict()}
    measured("exact_repeat_three_query_cache", lambda: enc.encode_queries(root, domain, QUERIES[0]), args.repeats)
    cache_evidence["query_after_repeat"] = {"frozen": enc._query_cached.cache_info()._asdict(), "adapted": enc._adapted_query_cached.cache_info()._asdict()}
    likes = [bundle["item_ids"][i] for i in (10, 100, 1000)]
    learned = measured("cold_three_user_heads", lambda: {mode: models.personalization(root, domain, mode, likes) for mode in models.MODES})
    measured("warm_three_user_heads", lambda: {mode: models.personalization(root, domain, mode, likes) for mode in models.MODES}, args.repeats)
    images, image_identities = catalog_images(root, bundle["items"])
    image_vector = measured("cold_image_encoder", lambda: enc.encode_image(root, images[0]))
    durations = []
    for image in images[1:]:
        started = time.perf_counter()
        enc.encode_image(root, image)
        durations.append((time.perf_counter() - started) * 1000)
    records.append({"stage": "warm_unique_image", **stats(durations)})
    # LRU capacity is eight; reinsert the exact image before cache-hit timing.
    enc.encode_image(root, images[0])
    cache_evidence["image_before_repeat"] = enc._image_cached.cache_info()._asdict()
    measured("exact_repeat_image_cache", lambda: enc.encode_image(root, images[0]), args.repeats)
    cache_evidence["image_after_repeat"] = enc._image_cached.cache_info()._asdict()

    parity = []
    for variant, extra in (("text", {}), ("multimodal_feedback", {"image_vector": image_vector, "positive_ids": likes}),
                           ("avoid_budget", {"avoid_terms": "leather; red", "max_price": 100.})):
        def rank(function):
            return {mode: function(bundle["items"], candidates[mode]["text"], bundle["image"],
                                   query_vector=queries[mode], k=12,
                                   **({**extra, "personalization_scores": learned[mode]["scores"]} if variant == "multimodal_feedback" else extra))
                    for mode in models.MODES}
        before, after = rank(baseline.rank_products), rank(studio.rank_products)
        if before != after:
            raise AssertionError(f"Complete-row ranking parity failed: {variant}")
        parity.append({"variant": variant, "identical_complete_rows": True,
                       "sha256": hashlib.sha256(json.dumps(after, sort_keys=True, allow_nan=False).encode()).hexdigest()})
        timings = {"before": [], "after": []}
        for repeat in range(args.repeats):
            # Alternate order to reduce monotonic thermal/cache bias.
            order = ("before", "after") if repeat % 2 == 0 else ("after", "before")
            for name in order:
                started = time.perf_counter()
                rank(baseline.rank_products if name == "before" else studio.rank_products)
                timings[name].append((time.perf_counter() - started) * 1000)
        for name in ("before", "after"):
            records.append({"stage": f"three_rankers_{variant}_{name}", **stats(timings[name])})

    # Warm engine path: encodes the three requests, image, three heads and ranks.
    # Unique-query timing above remains the honest model-forward request cost.
    def warm_engine(request=QUERIES[0]):
        catalog = enc.load_catalog(root, domain)
        q = enc.encode_queries(root, domain, request)
        im = enc.encode_image(root, images[0])
        for mode in models.MODES:
            candidate = models.candidate_catalog(root, domain, mode)
            user = models.personalization(root, domain, mode, likes)
            studio.rank_products(catalog["items"], candidate["text"], catalog["image"],
                                 query_vector=q[mode], image_vector=im, positive_ids=likes,
                                 personalization_scores=user["scores"], k=12)
    measured("warm_exact_repeat_engine", warm_engine, args.repeats)
    engine_queries = [query + " for casual daily use" for query in QUERIES]
    engine_durations = []
    for query in engine_queries:
        started = time.perf_counter()
        warm_engine(query)
        engine_durations.append((time.perf_counter() - started) * 1000)
    records.append({"stage": "warm_unique_request_engine", **stats(engine_durations)})
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    hardware = {}
    if sys.platform == "darwin":
        for key in ("machdep.cpu.brand_string", "hw.memsize"):
            response = subprocess.run(["sysctl", "-n", key], capture_output=True, text=True, check=False)
            if response.returncode == 0:
                hardware[key] = response.stdout.strip()
    payload = {"schema_version": 1, "complete": True, "benchmark_id": "studio_cpu_seed42",
               "environment": {"platform": platform.platform(), "machine": platform.machine(),
                               "processor": platform.processor(), "logical_cpus": os.cpu_count(), "hardware": hardware,
                               "torch": torch.__version__, "numpy": np.__version__, "torch_threads": 4, "thread_environment": thread_environment,
                               "device": "cpu", "mps_available": torch.backends.mps.is_available(),
                               "process_peak_rss_mib": peak / (1024**2 if sys.platform == "darwin" else 1024)},
               "protocol": {"domain": domain, "catalog_size": len(bundle["items"]), "models": list(models.MODES),
                            "repeats": args.repeats, "unique_query_count": len(unique), "unique_image_count": len(durations),
                            "queries": QUERIES, "engine_unique_queries": engine_queries, "image_samples": image_identities,
                            "baseline_ref": args.baseline_ref,
                            "baseline_source_sha256": hashlib.sha256(baseline_source.encode()).hexdigest(),
                            "optimized_source_sha256": hashlib.sha256((ROOT / "src/shoplens/studio.py").read_bytes()).hexdigest(),
                            "timing": "perf_counter wall time; CPU synchronous; paired alternating ranker order",
                            "background_compute": "Run without other benchmark/training processes"},
               "stages": records, "ranking_parity": parity, "cache_evidence": cache_evidence,
               "limitations": ["One local CPU environment and one selected seed/domain; timings are environment-specific.",
                               "Cold rows are one process-first observation, not a cold-start distribution; OS disk cache is not flushed.",
                               "Engine measurements exclude Streamlit rendering, browser, network and model downloads.",
                               "Exact repeats reuse request embeddings; unique requests still require real model forwards.",
                               "Small timing samples give descriptive p50/p95, not a latency service guarantee; each row records its actual count.",
                               "Unique engine requests include model forwards and ranking; exact-repeat engine requests reuse embeddings. No baseline end-to-end model-forward speedup is asserted.",
                               "Metadata optimization preserves the complete ranker rows; it does not change model quality."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "complete": True}), flush=True)


if __name__ == "__main__":
    main()
