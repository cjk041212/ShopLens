#!/usr/bin/env python3
"""Read-only ShopLens runtime checks; never load an LLM or access the network."""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import sys
from pathlib import Path

DOMAINS = ("baby", "sports", "clothing")


def _runtime_api(root):
    source = (root / "src").resolve()
    sys.path.insert(0, str(source))
    encoders = importlib.import_module("shoplens.studio_encoders")
    models = importlib.import_module("shoplens.studio_models")
    if not Path(encoders.__file__).resolve().is_relative_to(source) or not Path(models.__file__).resolve().is_relative_to(source):
        raise RuntimeError("Python imported ShopLens from another checkout. Use this project's virtual environment.")
    return encoders, models


def _photo_check(root, items):
    missing, unsafe = [], []
    for item in items:
        value = item.get("image_path")
        if not isinstance(value, str) or not value:
            missing.append(str(item["item_id"]))
            continue
        relative = Path(value)
        if relative.is_absolute() or ".." in relative.parts:
            unsafe.append(str(item["item_id"]))
            continue
        photo = (root / relative).resolve()
        if not photo.is_relative_to(root):
            unsafe.append(str(item["item_id"]))
        elif not photo.is_file():
            missing.append(str(item["item_id"]))
    return {"checked": len(items), "missing_count": len(missing), "unsafe_count": len(unsafe),
            "missing_examples": missing[:5], "unsafe_examples": unsafe[:5]}


def check(root, *, require_weights=False):
    root = Path(root).expanduser().resolve()
    report = {"ok": False, "root": str(root), "python": platform.python_version(),
              "scope": "Read-only catalog, photos, verified selected exports and local encoder-file presence. No LLM load, encoding, training or network.",
              "require_weights": require_weights, "domains": {}, "weights": None, "errors": [], "warnings": []}
    if sys.version_info < (3, 12):
        report["errors"].append("Python 3.12 or newer is required for the pinned release dependencies.")
        return report
    try:
        encoders, models = _runtime_api(root)
    except (ImportError, RuntimeError, OSError) as exc:
        report["errors"].append(f"Runtime dependencies or source package are unavailable: {exc}. Install requirements.txt and run python -m pip install -e . in this project.")
        return report
    for domain in DOMAINS:
        entry = {"catalog_ok": False, "photos": None, "models": {}}
        report["domains"][domain] = entry
        try:
            catalog = encoders.load_catalog(root, domain)
            entry.update(catalog_ok=True, items=len(catalog["items"]),
                         text_shape=list(catalog["text"].shape), image_shape=list(catalog["image"].shape))
            entry["photos"] = _photo_check(root, catalog["items"])
            if entry["photos"]["missing_count"] or entry["photos"]["unsafe_count"]:
                report["errors"].append(f"{domain}: local catalog photos are missing or use nonportable paths.")
        except (OSError, ValueError, RuntimeError, KeyError, ImportError) as exc:
            report["errors"].append(f"{domain}: catalog/base feature verification failed: {exc}")
        try:
            statuses = models.model_status(root, domain)
            for mode in models.MODES:
                if not statuses[mode].get("ready"):
                    raise ValueError(f"{mode}: {statuses[mode].get('error') or 'selected export is unavailable'}")
                # An empty history performs strict export verification, not an
                # invented recommendation. It never loads Qwen or an adapter.
                value = models.personalization(root, domain, mode, [])
                metadata = value["metadata"]
                if value["scores"] is not None or metadata.get("active") is not False:
                    raise ValueError(f"{mode}: empty-history inference contract failed")
                entry["models"][mode] = {"ready": True, "seed": metadata["seed"],
                                          "selected_epoch": metadata["selected_epoch"],
                                          "artifact_signature": metadata["artifact_signature"]}
        except (OSError, ValueError, RuntimeError, KeyError, ImportError) as exc:
            report["errors"].append(f"{domain}: selected-model export verification failed: {exc}")
    try:
        state = encoders.status(root)
        report["weights"] = {"ready": state["ready"], "query_ready": state["query_ready"],
                             "image_ready": state["image_ready"], "device": state["device"],
                             "qwen": {key: state["qwen"].get(key) for key in ("ready", "model_id", "revision", "error")},
                             "clip": {key: state["clip"].get(key) for key in ("ready", "model_id", "revision", "error")}}
        if not state["ready"]:
            message = "Pinned local encoder weights are not prepared. Run python scripts/prepare_studio_models.py --root . --include-clip, then rerun this check with --require-weights."
            report["errors" if require_weights else "warnings"].append(message)
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        report["errors"].append(f"Local encoder-file status check failed: {exc}")
    report["ok"] = not report["errors"]
    report["launch_command"] = "python -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501"
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--require-weights", action="store_true", help="Fail if pinned local Qwen/CLIP files are not prepared.")
    parser.add_argument("--json", action="store_true", help="Print the complete compact check report as JSON.")
    args = parser.parse_args()
    report = check(args.root, require_weights=args.require_weights)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print("ShopLens runtime check: " + ("PASS" if report["ok"] else "NOT READY"))
        for domain, details in report["domains"].items():
            print(f"  {domain}: catalog={details['catalog_ok']}, verified selected models={len(details['models'])}/3")
        for message in report["warnings"]:
            print("WARNING: " + message)
        for message in report["errors"]:
            print("ERROR: " + message)
        if report["ok"]:
            print("Start on loopback only: " + report["launch_command"])
            print("This check did not load the language model or verify fresh-query encoding speed.")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
