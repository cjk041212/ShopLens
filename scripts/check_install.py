#!/usr/bin/env python3
"""Read-only ShopLens runtime checks; never load an LLM or access the network."""

from __future__ import annotations

import argparse
import hashlib
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


def check_benchmarks(root):
    """Check the portable benchmark view without shopping assets or encoders."""
    root = Path(root).expanduser().resolve()
    report = {"ok": False, "scope": "Portable recorded benchmarks and image/model checksums; no inference, downloads or training.",
              "benchmarks": {}, "errors": []}
    sys.path.insert(0, str(root / "src"))
    try:
        from shoplens.search_benchmark import load_search_results
        from shoplens.fashion_retrieval import load_fashion_results
        from shoplens.recommendation_diagnostics import load_recommendation_diagnostics
        search = load_search_results(root)
        report["benchmarks"]["esci"] = {"queries": len(search["queries"]), "models": len(search["summary"]["models"])}
        fashion = load_fashion_results(root)
        summary = fashion["summary"]
        for relative, expected in (("assets/benchmarks/fashion/manifest.json", summary["manifest_sha256"]),
                                   ("assets/benchmarks/fashion/features.npz", summary["features_sha256"]),
                                   (summary["model_path"], summary["model_sha256"])):
            path = (root / relative).resolve()
            if not path.is_relative_to(root) or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError(f"FashionIQ checksum mismatch: {relative}")
        manifest = json.loads((root / "assets/benchmarks/fashion/manifest.json").read_text())
        for image in manifest["images"].values():
            path = (root / image["path"]).resolve()
            if not path.is_relative_to(root) or path.stat().st_size != image["bytes"] or hashlib.sha256(path.read_bytes()).hexdigest() != image["sha256"]:
                raise ValueError("FashionIQ photograph checksum mismatch")
        split_hashes = {phase: {manifest["images"][item_id]["sha256"]
                               for pool in pools.values() for item_id in pool}
                        for phase, pools in manifest["candidate_pools"].items()}
        if any(split_hashes[a] & split_hashes[b] for a, b in (("train", "dev"), ("train", "evaluation"), ("dev", "evaluation"))):
            raise ValueError("FashionIQ candidate splits contain byte-identical photographs")
        report["benchmarks"]["fashion"] = {"queries": len(fashion["queries"]), "verified_photos": len(manifest["images"]), "fusion_checkpoint": True}
        diagnostic = load_recommendation_diagnostics(root / "results/diagnostics/recommendation_diagnostics.json")
        report["benchmarks"]["diagnostics"] = {"history_records": len(diagnostic["history_curves"]), "checkpoints": len(diagnostic["exposure"])}
        performance = json.loads((root / "results/performance/studio.json").read_text())
        if performance.get("complete") is not True or len(performance.get("ranking_parity", [])) != 3 or not all(r.get("identical_complete_rows") is True for r in performance["ranking_parity"]):
            raise ValueError("Studio performance profile is incomplete or row parity failed")
        report["benchmarks"]["performance"] = {"measured_stages": len(performance["stages"]), "parity_cases": 3}
    except (OSError, ValueError, KeyError, TypeError, ImportError) as exc:
        report["errors"].append(str(exc))
    report["ok"] = not report["errors"]
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--require-weights", action="store_true", help="Fail if pinned local Qwen/CLIP files are not prepared.")
    parser.add_argument("--json", action="store_true", help="Print the complete compact check report as JSON.")
    parser.add_argument("--benchmarks-only", action="store_true", help="Verify recorded benchmark files and photos without restoring the shopping runtime.")
    args = parser.parse_args()
    report = check_benchmarks(args.root) if args.benchmarks_only else check(args.root, require_weights=args.require_weights)
    if args.json:
        print(json.dumps(report, indent=2))
    elif args.benchmarks_only:
        print("ShopLens portable benchmarks: " + ("PASS" if report["ok"] else "NOT READY"))
        for name, details in report["benchmarks"].items():
            print(f"  {name}: {json.dumps(details)}")
        for message in report["errors"]:
            print("ERROR: " + message)
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
