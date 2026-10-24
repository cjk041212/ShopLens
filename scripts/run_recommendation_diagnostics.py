#!/usr/bin/env python3
"""Measure fixed-model history/popularity diagnostics; no training or user labels."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True, help="Existing prepared interactions and seed42 exports; read only.")
    parser.add_argument("--output", type=Path, default=ROOT / "results/diagnostics/recommendation_diagnostics.json")
    parser.add_argument("--max-users", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if min(args.max_users, args.batch_size, args.bootstrap_repeats, args.threads) < 1:
        parser.error("All numerical options must be positive")
    import torch
    from shoplens.recommendation_diagnostics import run_diagnostics
    torch.set_num_threads(args.threads)
    started = time.monotonic()
    result = run_diagnostics(args.runtime_root, max_users=args.max_users, batch_size=args.batch_size,
                             bootstrap_repeats=args.bootstrap_repeats,
                             progress=lambda value: print(json.dumps({"unit_complete": value}), flush=True))
    result["execution"] = {"elapsed_seconds": time.monotonic() - started, "python": platform.python_version(),
                           "torch": torch.__version__, "platform": platform.system(), "architecture": platform.machine(),
                           "torch_threads": args.threads}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", dir=args.output.parent, prefix=".diagnostics-", delete=False, encoding="utf-8") as stream:
            temporary = Path(stream.name)
            json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, args.output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(json.dumps({"complete": True, "output": str(args.output), "history_records": len(result["history_curves"]),
                      "elapsed_seconds": result["execution"]["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
