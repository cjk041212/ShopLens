#!/usr/bin/env python3
"""Run the declared Qwen matrix sequentially on one allocated CUDA GPU."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domains", nargs="+", choices=("baby", "sports", "clothing"), default=["baby", "sports", "clothing"])
    parser.add_argument("--seeds", nargs="+", type=int, choices=(42, 43, 44), default=[42, 43, 44])
    parser.add_argument("--model-path", type=Path, default=ROOT / "data/models/Qwen3-Embedding-0.6B")
    parser.add_argument("--runtime-seconds", type=int, default=20400, help="Operational wall-clock budget, not an experimental hyperparameter")
    args = parser.parse_args()
    if not os.environ.get("PBS_JOBID"):
        raise RuntimeError("Real GPU work requires an allocated PBS job; do not run this on a login node")
    import torch
    from shoplens.models import digest
    from shoplens.qwen_pipeline import DEFAULT_CONFIG, ResumableStop, run_one
    from shoplens.qwen_text import MODEL_ID, REVISION
    from shoplens.qwen_evaluate import _atomic_json
    if not torch.cuda.is_available():
        raise RuntimeError("Allocated job has no available CUDA device")
    source = json.loads((ROOT / "experiments/qwen/model_source.json").read_text())
    if source["model_id"] != MODEL_ID or source["revision"] != REVISION:
        raise ValueError("Downloaded model revision differs from the pinned experiment")
    manifest = json.loads((args.model_path / "shoplens_source.json").read_text())
    if manifest["revision"] != REVISION:
        raise ValueError("Local model bundle revision mismatch")
    for name, expected in manifest["sha256"].items():
        if digest(args.model_path / name) != expected:
            raise ValueError(f"Local model bundle file changed: {name}")
    torch.set_num_threads(min(8, max(1, int(os.environ.get("OMP_NUM_THREADS", "4")))))
    started = time.monotonic()
    job_id = os.environ["PBS_JOBID"]
    job_dir = ROOT / "experiments/qwen/jobs" / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    environment = {"job_id": job_id, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   "python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda,
                   "gpu": torch.cuda.get_device_name(0), "model_id": MODEL_ID, "revision": REVISION,
                   "packages": {name: importlib.metadata.version(name) for name in
                                ("transformers", "peft", "accelerate", "numpy")},
                   "domains": args.domains, "seeds": args.seeds, "config": DEFAULT_CONFIG,
                   "code_sha256": {path.name: digest(path) for path in sorted((ROOT / "src/shoplens").glob("qwen_*.py"))},
                   "runtime_seconds": args.runtime_seconds}
    _atomic_json(job_dir / "environment.json", environment)
    progress = {"job_id": job_id, "status": "running", "completed_units": [], "current": None}
    _atomic_json(job_dir / "status.json", progress)
    print("QWEN_ENVIRONMENT", json.dumps(environment), flush=True)
    try:
        for domain in args.domains:
            for seed in args.seeds:
                remaining = args.runtime_seconds - (time.monotonic() - started)
                if remaining < 120:
                    raise ResumableStop("Job time budget exhausted before next unit")
                progress["current"] = {"domain": domain, "seed": seed}
                _atomic_json(job_dir / "status.json", progress)
                config = dict(DEFAULT_CONFIG)
                config["max_runtime_seconds"] = remaining
                print("QWEN_UNIT_START", domain, seed, "remaining_seconds", round(remaining), flush=True)
                run_one(ROOT, domain, seed, args.model_path, device="cuda", config=config)
                for mode in ("qwen_frozen", "qwen_semantic", "qwen_preference"):
                    complete = ROOT / "artifacts/qwen" / domain / mode / f"seed{seed}" / "complete.json"
                    if not complete.is_file():
                        raise RuntimeError(f"run_one returned without a completed formal unit: {domain}/{mode}/{seed}")
                    progress["completed_units"].append({"domain": domain, "seed": seed, "model": mode})
                _atomic_json(job_dir / "status.json", progress)
                print("QWEN_UNIT_COMPLETE", domain, seed, "elapsed_seconds", round(time.monotonic() - started, 1), flush=True)
        progress.update(status="complete", current=None)
    except ResumableStop as exc:
        progress.update(status="incomplete_resumable", reason=str(exc))
        print("QWEN_RESUMABLE_STOP", str(exc), flush=True)
    except Exception:
        progress.update(status="failed", error=traceback.format_exc())
        raise
    finally:
        progress["elapsed_seconds"] = time.monotonic() - started
        progress["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        _atomic_json(job_dir / "status.json", progress)
    print("QWEN_JOB_STATUS", json.dumps(progress), flush=True)


if __name__ == "__main__":
    main()
