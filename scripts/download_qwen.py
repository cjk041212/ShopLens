#!/usr/bin/env python3
"""Download and fingerprint the public, pinned Qwen model; no GPU computation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from huggingface_hub import snapshot_download
    snapshot = Path(snapshot_download(MODEL_ID, revision=REVISION, cache_dir=str(root / "data/hf-cache"),
                                     token=False, local_files_only=args.local_files_only,
                                     allow_patterns=["*.json", "*.safetensors", "tokenizer*", "vocab*", "merges*", "*.model", "README.md"]))
    output = root / "data/models/Qwen3-Embedding-0.6B"
    output.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for source in sorted(snapshot.iterdir()):
        if not source.is_file():
            continue
        destination = output / source.name
        if not destination.exists():
            shutil.copy2(source, destination)
        with source.open("rb") as stream:
            source_hash = hashlib.file_digest(stream, "sha256").hexdigest()
        with destination.open("rb") as stream:
            destination_hash = hashlib.file_digest(stream, "sha256").hexdigest()
        if destination_hash != source_hash:
            raise ValueError(f"Existing model file differs from pinned source: {source.name}")
        hashes[source.name] = source_hash
    manifest = {"model_id": MODEL_ID, "revision": REVISION, "sha256": hashes}
    (output / "shoplens_source.json").write_text(json.dumps(manifest, indent=2) + "\n")
    source_record = {"model_id": MODEL_ID, "revision": REVISION, "public_download": True,
                     "snapshot": str(snapshot), "output": str(output.relative_to(root)), "sha256": hashes}
    record_path = root / "experiments/qwen/model_source.json"
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(json.dumps(source_record, indent=2) + "\n")
    print(json.dumps({"model_id": MODEL_ID, "revision": REVISION, "output": str(output), "files": len(hashes)}))


if __name__ == "__main__":
    main()
