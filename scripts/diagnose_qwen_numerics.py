#!/usr/bin/env python3
"""Isolated CUDA numerical diagnosis; never writes formal model artifacts/results."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import copy
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def scalar(value):
    value = float(value)
    return value if math.isfinite(value) else str(value)


def tensor_stats(value):
    detached = value.detach()
    finite = torch.isfinite(detached)
    usable = detached[finite].double()
    return {"shape": list(detached.shape), "dtype": str(detached.dtype),
            "finite": bool(finite.all()), "nonfinite_count": int((~finite).sum()),
            "max_abs": scalar(usable.abs().max()) if usable.numel() else None,
            "norm_fp64": scalar(usable.square().sum().sqrt()) if usable.numel() else None}


def gradient_stats(named_parameters):
    nonfinite, missing, max_abs, sum_squares = [], [], 0., 0.
    for name, parameter in named_parameters:
        if parameter.grad is None:
            missing.append(name)
            continue
        stats = tensor_stats(parameter.grad)
        if not stats["finite"]:
            nonfinite.append(name)
        if stats["max_abs"] is not None:
            max_abs = max(max_abs, float(stats["max_abs"]))
            sum_squares += float(stats["norm_fp64"]) ** 2
    return {"finite": not nonfinite, "nonfinite_parameter_names": nonfinite[:30],
            "nonfinite_parameter_count": len(nonfinite), "missing_gradient_count": len(missing),
            "max_abs": scalar(max_abs), "norm_fp64": scalar(math.sqrt(sum_squares))}


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(part) for key, part in value.items()}
    if isinstance(value, list):
        return [cpu_tree(part) for part in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(part) for part in value)
    return copy.deepcopy(value)


def backend_context(backend):
    if backend == "default":
        return nullcontext()
    from torch.nn.attention import SDPBackend, sdpa_kernel
    return sdpa_kernel(SDPBackend.MATH if backend == "math" else SDPBackend.FLASH_ATTENTION)


def live_embeddings(encoder, tokenizer, texts, device, max_length):
    from shoplens.qwen_text import last_token_pool
    batch = tokenizer(list(texts), padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    batch = {key: value.to(device) for key, value in batch.items()}
    output = encoder(**batch, return_dict=True, use_cache=False)
    pooled = last_token_pool(output.last_hidden_state, batch["attention_mask"]).float()
    norms = pooled.detach().double().norm(dim=-1)
    pool_stats = tensor_stats(pooled)
    pool_stats.update(min_row_norm=scalar(norms.min()), max_row_norm=scalar(norms.max()),
                      token_length_min=int(batch["attention_mask"].sum(1).min()),
                      token_length_max=int(batch["attention_mask"].sum(1).max()))
    return F.normalize(pooled, dim=-1), pool_stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--backend", choices=("default", "math", "flash"), default="default")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--runtime-seconds", type=int, default=780)
    parser.add_argument("--replay-state", type=Path, help="Recreate exactly one saved batch/state; never optimizer.step")
    args = parser.parse_args()
    if not os.environ.get("PBS_JOBID") or not torch.cuda.is_available():
        raise RuntimeError("Numerical GPU diagnosis requires PBS_JOBID and allocated CUDA; no login-node GPU work")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("This diagnostic requires exactly one visible allocated GPU")
    if not 1 <= args.steps <= 100 or args.runtime_seconds < 1:
        raise ValueError("steps must be 1..100 and runtime-seconds positive")
    root = args.root.resolve()
    model_path = args.model_path or root / "data/models/Qwen3-Embedding-0.6B"
    output = root / "experiments/qwen/diagnostics" / os.environ["PBS_JOBID"]
    output.mkdir(parents=True, exist_ok=True)
    tag = ("replay" if args.replay_state else "prefix") + "_" + args.backend
    summary_path, steps_path = output / f"{tag}_summary.json", output / f"{tag}_steps.jsonl"
    if summary_path.exists() or steps_path.exists():
        raise FileExistsError("Diagnostic evidence already exists; preserve it and use a new PBS job")

    from shoplens.models import digest, seed_everything
    from shoplens.qwen_data import build_data
    from shoplens.qwen_models import multi_positive_infonce
    from shoplens.qwen_pipeline import DEFAULT_CONFIG, _batch_arrays, _cpu_state, _make_ranker, _restore_rng, _rng_state
    from shoplens.qwen_text import (MODEL_ID, REVISION, adapter_parameters, adapter_state,
                                   load_text_model, restore_adapter)
    torch.set_num_threads(min(8, max(1, int(os.environ.get("OMP_NUM_THREADS", "4")))))
    started = time.monotonic()
    config = copy.deepcopy(DEFAULT_CONFIG)
    data = build_data(root, "clothing", 42, config["adaptation_limit"], config["max_history"])
    base_dir = root / "artifacts/qwen/clothing/base"
    shared = root / "artifacts/qwen/clothing/warmup/seed42"
    sources = {"base": base_dir / "features.npz", "warmup": shared / "selected.pt",
               "sampling": shared / "sampling.npz", "anchors": shared / "review_anchors.npz"}
    base_meta = json.loads((base_dir / "features.json").read_text())
    if digest(sources["base"]) != base_meta["features_sha256"]:
        raise ValueError("Base cache checksum mismatch")
    with np.load(sources["base"], allow_pickle=False) as values:
        if values["item_ids"].astype(str).tolist() != data.item_ids:
            raise ValueError("Base catalog order differs")
        base_text, images = values["text"].copy(), values["image"].copy()
    with np.load(sources["sampling"], allow_pickle=False) as values:
        if values["case_ids"].astype(str).tolist() != [row["case_id"] for row in data.adaptation_examples]:
            raise ValueError("Shared case order differs")
        orders, negatives = values["orders"].copy(), values["negatives"].copy()
    with np.load(sources["anchors"], allow_pickle=False) as values:
        if values["case_ids"].astype(str).tolist() != [row["case_id"] for row in data.adaptation_examples]:
            raise ValueError("Anchor case order differs")
        anchors = torch.as_tensor(values["anchors"].copy(), device="cuda")
    # Match _train_mode construction and RNG consumption exactly; no run_one/formal write calls.
    seed_everything(42)
    ranker = _make_ranker(base_text, images, config, "cuda")
    ranker.load_state_dict(torch.load(sources["warmup"], map_location="cuda", weights_only=False)["state_dict"])
    encoder, tokenizer = load_text_model(model_path, "cuda", create_lora=True, lora_config=config["lora_config"])
    parameters = adapter_parameters(encoder)
    head_parameters = list(ranker.parameters())
    optimizer = torch.optim.AdamW([
        {"params": head_parameters, "lr": config["head_lr"], "weight_decay": config["head_weight_decay"]},
        {"params": parameters, "lr": config["lora_lr"], "weight_decay": config["lora_weight_decay"]}])
    replay = None
    if args.replay_state:
        replay = torch.load(args.replay_state, map_location="cuda", weights_only=False)
        if not replay.get("diagnostic") or replay.get("domain") != "clothing" or replay.get("seed") != 42:
            raise ValueError("Replay input is not a Clothing42 numerical diagnostic checkpoint")
        ranker.load_state_dict(replay["head"])
        restore_adapter(encoder, replay["adapter"])
        optimizer.load_state_dict(replay["optimizer"])
        _restore_rng(replay["rng"])
    ranker.train(); encoder.train()
    head_named = list(ranker.named_parameters())
    lora_named = [(name, value) for name, value in encoder.named_parameters() if value.requires_grad]
    summary = {"diagnostic": True, "formal_results": False, "domain": "clothing", "seed": 42,
               "mode": "qwen_semantic", "backend": args.backend, "replay": bool(replay),
               "job_id": os.environ["PBS_JOBID"], "model_id": MODEL_ID, "revision": REVISION,
               "torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
               "config": config, "input_sha256": {key: digest(path) for key, path in sources.items()},
               "code_sha256": {path.name: digest(path) for path in sorted((root / "src/shoplens").glob("qwen_*.py"))},
               "script_sha256": digest(Path(__file__)), "anchors": tensor_stats(anchors),
               "status": "running", "successful_optimizer_steps": 0}
    if args.replay_state:
        summary["replay_source_sha256"] = digest(args.replay_state)
    print("QWEN_NUMERICS_START", json.dumps(summary, allow_nan=False), flush=True)
    active_snapshot = None
    row = None
    try:
        for iteration in range(1 if replay else args.steps):
            if time.monotonic() - started >= args.runtime_seconds:
                summary["status"] = "diagnostic_time_limit"
                break
            start = int(replay["batch"]["start"]) if replay else iteration * config["adaptation_batch_size"]
            positions, histories, positive, negative = _batch_arrays(data.adaptation_examples, orders[0], negatives[0], start,
                config["adaptation_batch_size"], "cuda", config["max_history"])
            batch_info = {"epoch": 1, "start": start, "step": start // config["adaptation_batch_size"] + 1,
                          "positions": positions.tolist(), "case_ids": [data.adaptation_examples[int(i)]["case_id"] for i in positions],
                          "positive_indices": positive.cpu().tolist(), "negative_indices": negative.cpu().tolist(),
                          "histories": histories.cpu().tolist()}
            if replay and batch_info != replay["batch"]:
                raise ValueError("Replay batch no longer matches shared order/data")
            optimizer.zero_grad(set_to_none=True)
            # Capture the clean pre-forward state on CPU. A failed backward/update cannot overwrite this evidence.
            active_snapshot = {"diagnostic": True, "domain": "clothing", "seed": 42, "backend": args.backend,
                               "head": _cpu_state(ranker), "adapter": adapter_state(encoder),
                               "optimizer": cpu_tree(optimizer.state_dict()), "rng": _rng_state(), "batch": batch_info,
                               "input_sha256": summary["input_sha256"], "config": config}
            row = {"diagnostic": True, "step": batch_info["step"], "start": start, "backend": args.backend,
                   "elapsed_seconds": round(time.monotonic() - started, 3)}
            anchor = anchors[torch.as_tensor(positions, device="cuda")]
            with backend_context(args.backend):
                concatenated = torch.cat((positive, negative))
                unique, inverse = torch.unique(concatenated, sorted=True, return_inverse=True)
                live, pool = live_embeddings(encoder, tokenizer, [data.texts[int(i)] for i in unique.cpu()], "cuda", config["max_length"])
                positive_text, negative_text = live[inverse[:len(positions)]], live[inverse[len(positions):]]
                bpr = ranker(histories, positive, negative, positive_text.detach(), negative_text.detach())
                semantic = multi_positive_infonce(anchor, positive_text, positive, config["semantic_temperature"])
                loss = bpr + semantic
                row.update(anchors=tensor_stats(anchor), live=tensor_stats(live), pooled=pool,
                           bpr=scalar(bpr.detach()), infonce=scalar(semantic.detach()), loss=scalar(loss.detach()))
                row["forward_finite"] = all((row["anchors"]["finite"], row["live"]["finite"], pool["finite"],
                                              bool(torch.isfinite(bpr)), bool(torch.isfinite(semantic)), bool(torch.isfinite(loss))))
                if not row["forward_finite"]:
                    row["anomaly"] = "nonfinite_forward"
                else:
                    # Formal semantic path probes its detached BPR on its first successful step.
                    if batch_info["step"] == 1:
                        probe = torch.autograd.grad(bpr, parameters, retain_graph=True, allow_unused=True)
                        row["bpr_to_lora_all_unused"] = all(value is None for value in probe)
                    loss.backward()
                    row["head_gradients"] = gradient_stats(head_named)
                    row["lora_gradients"] = gradient_stats(lora_named)
                    if not row["head_gradients"]["finite"] or not row["lora_gradients"]["finite"]:
                        row["anomaly"] = "nonfinite_gradient_before_clip"
                    else:
                        # Same thresholds as formal pipeline, with a diagnostic guard before any update.
                        head_norm = torch.nn.utils.clip_grad_norm_(head_parameters, 5., error_if_nonfinite=True)
                        lora_norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
                        row.update(head_clip_input_norm=scalar(head_norm), lora_clip_input_norm=scalar(lora_norm))
                        if not replay:
                            optimizer.step()
                            bad = [name for name, value in head_named + lora_named if not bool(torch.isfinite(value).all())]
                            if bad:
                                row.update(anomaly="nonfinite_parameter_after_optimizer", nonfinite_parameter_names=bad[:30])
                            else:
                                summary["successful_optimizer_steps"] += 1
            with steps_path.open("a") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            if row.get("anomaly"):
                summary.update(status="diagnostic_anomaly", first_anomaly=row)
                break
            if replay or batch_info["step"] % 10 == 0:
                print("QWEN_NUMERICS_STEP", json.dumps(row, allow_nan=False), flush=True)
        else:
            summary["status"] = "diagnostic_replay_finite" if replay else "diagnostic_prefix_finite"
    except Exception:
        summary.update(status="diagnostic_exception", error=traceback.format_exc(), first_anomaly=row)
    finally:
        if summary["status"] in {"diagnostic_anomaly", "diagnostic_exception"} and active_snapshot is not None:
            state_path = output / f"{tag}_clean_before_failure.pt"
            torch.save(active_snapshot, state_path)
            summary.update(clean_state_path=str(state_path.relative_to(root)), clean_state_sha256=digest(state_path))
        summary["elapsed_seconds"] = time.monotonic() - started
        summary["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
        summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
        print("QWEN_NUMERICS_STATUS", json.dumps(summary, allow_nan=False), flush=True)
    return 2 if summary["status"] == "diagnostic_exception" else 0


if __name__ == "__main__":
    raise SystemExit(main())
