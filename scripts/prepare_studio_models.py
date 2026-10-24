#!/usr/bin/env python3
"""Explicitly prepare public, pinned Shopping Studio models outside formal runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from shoplens.studio_encoders import CLIP_ID, CLIP_REVISION, QWEN_ID, QWEN_REVISION, status


def _sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(root, *, offline=False, include_clip=False):
    from huggingface_hub import snapshot_download

    root = Path(root).expanduser().resolve()
    cache = root / "data/studio_hf_cache"
    patterns = ["config.json", "*.safetensors", "*.safetensors.index.json", "tokenizer.json", "tokenizer_config.json",
                "vocab.json", "merges.txt", "special_tokens_map.json", "added_tokens.json"]
    current = status(root)
    if not current["query_ready"]:
        snapshot_download(repo_id=QWEN_ID, revision=QWEN_REVISION, cache_dir=str(cache),
                          allow_patterns=patterns, token=False, local_files_only=offline)
    if include_clip and not current["image_ready"]:
        snapshot_download(repo_id=CLIP_ID, revision=CLIP_REVISION, cache_dir=str(cache),
                          allow_patterns=[*patterns, "preprocessor_config.json", "pytorch_model.bin"],
                          token=False, local_files_only=offline)
    current = status(root)
    if not current["query_ready"] or include_clip and not current["image_ready"]:
        raise RuntimeError("Pinned Studio models remain incomplete; check the explicit download error or local files.")
    manifest = {"schema_version": 1, "purpose": "Local Shopping Studio only; not formal experiment provenance.",
                "status": current, "model_files": {}}
    for kind in ("qwen", "clip"):
        if not current[kind]["ready"]:
            continue
        snapshot = Path(current[kind]["path"])
        manifest["model_files"][kind] = {
            path.name: {"sha256": _sha(path), "bytes": path.stat().st_size}
            for path in sorted(snapshot.iterdir()) if path.is_file()
        }
    destination = root / "data/studio_models/manifest.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {"manifest": str(destination), **current}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--offline", action="store_true", help="Use local pinned files only; never access the network.")
    parser.add_argument("--include-clip", action="store_true", help="Also download pinned CLIP if neither project nor existing user cache has it.")
    args = parser.parse_args()
    print(json.dumps(prepare(args.root, offline=args.offline, include_clip=args.include_clip), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
