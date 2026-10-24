#!/usr/bin/env python3
"""Measure local exported-tower serving and check it against the fitted head."""

from __future__ import annotations

import argparse
import gc
import json
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--domains", nargs="+", default=["baby", "sports", "clothing"])
    parser.add_argument("--models", nargs="+", choices=("qwen_frozen", "qwen_semantic", "qwen_preference"),
                        default=["qwen_frozen", "qwen_semantic", "qwen_preference"])
    parser.add_argument("--repeats", type=int, default=50)
    args = parser.parse_args()
    if args.repeats < 2:
        raise ValueError("At least two timing repetitions required")
    import numpy as np
    import torch
    from shoplens.models import load_items, pad_histories, stable_topk
    from shoplens.qwen_inference import QWEN_MODES, _cached_scorer, available_models, load_scorer, recommend
    from shoplens.qwen_models import QwenHistoryRecommender
    root = args.root.resolve()
    torch.set_num_threads(4)
    records = []
    for domain in args.domains:
        processed = root / "data/processed" / domain
        ids = [item["item_id"] for item in load_items(processed)]
        liked = [ids[index] for index in np.linspace(0, len(ids) - 1, 3, dtype=int)]
        for mode in args.models:
            unit = root / "artifacts/qwen" / domain / mode / "seed42"
            if mode not in available_models(unit):
                raise FileNotFoundError(f"Formal seed42 export is not ready: {domain}/{mode}")
            _cached_scorer.cache_clear()
            gc.collect()
            started = time.perf_counter()
            recommend(processed, unit, mode, liked, k=10)
            cold_ms = (time.perf_counter() - started) * 1000
            durations = []
            for _ in range(args.repeats):
                started = time.perf_counter()
                rows = recommend(processed, unit, mode, liked, k=10)
                durations.append((time.perf_counter() - started) * 1000)
                if len(rows) != 10 or set(liked) & {row["item_id"] for row in rows}:
                    raise ValueError("Serving returned selected likes or fewer than ten candidates")
            scorer = load_scorer(processed, unit, mode)
            with np.load(unit.parent.parent / "base/features.npz", allow_pickle=False) as values:
                base, image = values["text"].copy(), values["image"].copy()
            with np.load(unit / "candidate.npz", allow_pickle=False) as values:
                candidate = values["text"].copy()
            network = QwenHistoryRecommender(base, image, candidate, embedding_dim=scorer.metadata["embedding_dim"], max_history=scorer.max_history)
            state = torch.load(unit / "models" / f"{mode}.pt", map_location="cpu", weights_only=True)["state_dict"]
            network.load_state_dict(state)
            network.eval()
            histories = [[scorer.index[item] for item in reversed(liked)], [0], list(range(5, 0, -1))]
            with torch.inference_mode():
                expected = (network.user_vectors(torch.from_numpy(pad_histories(histories, scorer.max_history))) @ network.item_vectors().T).numpy()
            actual = scorer.score_histories(histories)
            error = float(np.max(np.abs(actual - expected)))
            same_top10 = all(np.array_equal(stable_topk(first, 10, history), stable_topk(second, 10, history))
                             for first, second, history in zip(actual, expected, histories))
            if error > 1e-5 or not same_top10:
                raise ValueError(f"Actual fitted-head/export parity failed: {domain}/{mode}, {error}, {same_top10}")
            record = {"domain": domain, "model": mode, "seed": 42, "device": "cpu",
                      "n_catalog_items": len(ids), "n_liked_items": len(liked), "repeats": args.repeats,
                      "cold_model_load_and_recommend_ms": cold_ms,
                      "warm_median_ms": float(np.median(durations)), "warm_p95_ms": float(np.quantile(durations, .95)),
                      "fitted_head_export_max_abs_error": error, "parity_top10_equal": same_top10,
                      "parity_histories": len(histories), "serving_sha256": scorer.metadata["serving_sha256"]}
            records.append(record)
            print(json.dumps(record), flush=True)
            del network, state, base, image, candidate, expected, actual, scorer
            gc.collect()
    hardware = {}
    if sys.platform == "darwin":
        for key in ("machdep.cpu.brand_string", "hw.memsize"):
            result = subprocess.run(["sysctl", "-n", key], capture_output=True, text=True, check=True)
            hardware[key] = result.stdout.strip()
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    output = {"platform": platform.platform(), "architecture": platform.machine(), "hardware": hardware,
              "complete_nine_model_matrix": len(records) == 9 and set(args.domains) == {"baby", "sports", "clothing"} and set(args.models) == set(QWEN_MODES),
              "torch": torch.__version__, "torch_threads": 4,
              "process_peak_rss_mib": peak / (1024 ** 2 if sys.platform == "darwin" else 1024),
              "definition": "CPU recommendation API with three selected catalog products and top10 output. Warm timings reuse the scorer; cold includes catalog and selected export loading/checksums after Python/Torch imports. Excludes browser, network, and Qwen text encoding. Process peak includes separate actual-head parity verification, not just serving.",
              "records": records}
    path = root / "artifacts/qwen/local_benchmark.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
