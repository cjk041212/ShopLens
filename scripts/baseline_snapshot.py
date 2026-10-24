#!/usr/bin/env python3
"""Hash an existing ShopLens baseline and verify that an upgrade preserved it.

This reads bytes only. It does not load pickle/PyTorch/NPZ files, copy weights,
modify an experiment, follow symlinks, access a network, or run training.

Before an upgrade (choose a new snapshot filename outside protected data):
  python scripts/baseline_snapshot.py snapshot --root /path/to/NLP-Coursework \
    --output /path/to/baseline-before-qwen.json
After the upgrade:
  python scripts/baseline_snapshot.py verify --root /path/to/NLP-Coursework \
    --snapshot /path/to/baseline-before-qwen.json

Added files are reported, but do not invalidate unchanged baseline bytes unless
--fail-on-new is supplied. A snapshot proves byte equality at verification time;
it is not a backup or a record of every intervening action.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import sys

DOMAINS = ("baby", "sports", "clothing")
DATA_FILES = ("items.jsonl", "events.jsonl", "reviews_train.jsonl", "audit.json", "images_audit.json")
ARTIFACT_FILES = ("results.json", "features.npz", "features.json")
FIXED_FILES = ("README.md", "requirements.txt", "requirements-lock-macos.txt", "pyproject.toml", "app.py",
               "experiments/baseline_budget.json", "experiments/complete.json",
               "deliverables/results_summary.json", "deliverables/results_summary.csv", "deliverables/实验报告.md")
IGNORED_PARTS = {"__pycache__", ".git", ".venv", "node_modules"}


class SnapshotError(ValueError):
    pass


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def stable_file_hash(path: Path) -> dict:
    """Reject a file that changes during reading instead of freezing a mixed state."""
    first = path.lstat()
    if not stat.S_ISREG(first.st_mode):
        raise SnapshotError("Selected entry is not a regular file; symlinks are not allowed")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    with os.fdopen(descriptor, "rb") as stream:
        opened = os.fstat(stream.fileno())
        if (first.st_dev, first.st_ino) != (opened.st_dev, opened.st_ino):
            raise SnapshotError("File changed before hashing")
        digest = hashlib.sha256()
        total = 0
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            total += len(block)
            digest.update(block)
        last = os.fstat(stream.fileno())
    current = path.lstat()
    identity = lambda value: (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)
    if identity(first) != identity(last) or identity(last) != identity(current) or total != last.st_size:
        raise SnapshotError("File changed during hashing; wait for baseline writes to finish")
    return {"sha256": digest.hexdigest(), "bytes": total}


def checked_path(root: Path, relative: str) -> Path:
    posix = PurePosixPath(relative)
    if posix.is_absolute() or not posix.parts or ".." in posix.parts or "\\" in relative:
        raise SnapshotError("Snapshot contains a nonrelative or unsafe filename")
    path = root.joinpath(*posix.parts)
    if not path.resolve(strict=False).is_relative_to(root):
        raise SnapshotError("Snapshot filename escapes the baseline root")
    if any(parent.is_symlink() for parent in (path, *path.parents) if parent != root.parent):
        raise SnapshotError("Snapshot filename traverses a symlink")
    return path


def eligible(path: Path, root: Path) -> bool:
    relative = path.relative_to(root)
    return not any(part in IGNORED_PARTS or part.startswith("._") or part.endswith(".egg-info")
                   for part in relative.parts)


def selected_paths(root: Path, *, include_source: bool = True) -> list[Path]:
    selected: set[Path] = set()
    selected.update(root / name for name in FIXED_FILES if (root / name).is_file())
    for domain in DOMAINS:
        for name in DATA_FILES:
            path = root / "data/processed" / domain / name
            if path.is_file():
                selected.add(path)
        for name in ARTIFACT_FILES:
            path = root / "artifacts" / domain / name
            if path.is_file():
                selected.add(path)
        for directory in (root / "artifacts" / domain / "models",
                          root / "artifacts" / domain / "text_adaptation",
                          root / "artifacts/budget_check" / domain,
                          root / "experiments/runs" / domain):
            selected.update(path for path in directory.rglob("*") if path.is_file() and eligible(path, root))
    selected.update(path for path in (root / "data/raw").glob("*.source.json") if path.is_file())
    if include_source:
        for directory, suffixes in (("src", {".py"}), ("tests", {".py"}),
                                    ("scripts", {".py", ".sh", ".pbs"})):
            selected.update(path for path in (root / directory).rglob("*")
                            if path.is_file() and path.suffix in suffixes and eligible(path, root))
    return sorted(selected, key=lambda path: path.relative_to(root).as_posix())


def baseline_status(root: Path) -> dict:
    output = {}
    expected_methods = {"popularity", "mf", "tfidf", "text", "multimodal", "adapted"}
    for domain in DOMAINS:
        directory = root / "artifacts" / domain
        path = directory / "results.json"
        rows = json.loads(path.read_text()).get("results", []) if path.exists() else []
        primary = [row for row in rows if row.get("split") == "test"]
        found = {(row.get("model"), row.get("seed")) for row in primary}
        expected = {(model, seed) for model in expected_methods for seed in (42, 43, 44)}
        output[domain] = {"test_records": len(primary), "expected_test_records": 18,
                          "expected_keys_complete": expected <= found,
                          "features_present": (directory / "features.npz").is_file(),
                          "model_files": sum(path.is_file() for path in (directory / "models").glob("*")),
                          "processed_present": {name: (root / "data/processed" / domain / name).is_file()
                                                for name in DATA_FILES}}
    return output


def snapshot(root: Path, *, include_source: bool = True) -> dict:
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise SnapshotError("Baseline root must be a directory")
    files = {}
    for path in selected_paths(root, include_source=include_source):
        relative = path.relative_to(root).as_posix()
        checked = checked_path(root, relative)
        try:
            files[relative] = stable_file_hash(checked)
        except (OSError, SnapshotError) as exc:
            raise SnapshotError(f"Cannot snapshot {relative}: {exc}") from exc
    if not files:
        raise SnapshotError("No baseline files found; verify the project root")
    document = {"schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "baseline_name": root.name, "hash_algorithm": "sha256",
                "include_source": include_source, "domains": list(DOMAINS),
                "baseline_status": baseline_status(root), "files": files,
                "scope_notes": [
                    "Tracks processed inputs, three-domain baseline results/features/models, adaptation checkpoints, per-seed runs, budget checks, source sidecars, and selected source/report files.",
                    "Raw review gzip files, product image bytes, HF caches, virtual environments, credentials, and logs are not read or covered.",
                    "Stored paths are relative; the snapshot contains hashes and sizes, never file contents or private root/account paths.",
                    "A hash snapshot is not a backup. Keep baseline directories intact and write Qwen features/models/results into separate directories.",
                ]}
    document["snapshot_sha256"] = hashlib.sha256(canonical(document)).hexdigest()
    return document


def write_snapshot(document: dict, destination: Path, root: Path) -> None:
    destination = destination.resolve(strict=False)
    existing = {path.resolve() for path in selected_paths(root.resolve(), include_source=True)}
    if destination in existing:
        raise SnapshotError("Snapshot output would overwrite a baseline file")
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents silently replacing the original preservation proof.
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(document, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def verify(root: Path, document: dict, *, fail_on_new: bool = False) -> dict:
    root = root.resolve(strict=True)
    if document.get("schema_version") != 1 or document.get("hash_algorithm") != "sha256":
        raise SnapshotError("Unsupported snapshot schema or hash algorithm")
    recorded = document.get("snapshot_sha256")
    content = {key: value for key, value in document.items() if key != "snapshot_sha256"}
    if recorded != hashlib.sha256(canonical(content)).hexdigest():
        raise SnapshotError("Snapshot JSON integrity check failed")
    files = document.get("files")
    if not isinstance(files, dict) or not files:
        raise SnapshotError("Snapshot does not contain a file inventory")
    missing, changed, errors = [], [], []
    matched = 0
    for relative, expected in files.items():
        path = checked_path(root, relative)
        if not path.exists():
            missing.append(relative)
            continue
        try:
            actual = stable_file_hash(path)
            if actual == expected:
                matched += 1
            else:
                changed.append({"path": relative, "expected_sha256": expected["sha256"],
                                "actual_sha256": actual["sha256"], "expected_bytes": expected["bytes"],
                                "actual_bytes": actual["bytes"]})
        except (OSError, SnapshotError) as exc:
            errors.append({"path": relative, "error": str(exc)})
    present = {path.relative_to(root).as_posix()
               for path in selected_paths(root, include_source=document.get("include_source", True))}
    added = sorted(present - files.keys())
    return {"unchanged": not missing and not changed and not errors and (not fail_on_new or not added),
            "matched_files": matched, "snapshot_files": len(files), "missing": missing,
            "changed": changed, "errors": errors, "added_in_baseline_scope": added,
            "fail_on_new": fail_on_new, "snapshot_sha256": recorded}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    make = commands.add_parser("snapshot", help="Read the baseline and write a new hash inventory")
    make.add_argument("--root", type=Path, required=True)
    make.add_argument("--output", type=Path, required=True)
    make.add_argument("--no-source", action="store_true", help="Omit Python/scripts/tests from preservation scope")
    check = commands.add_parser("verify", help="Rehash baseline files and report any change")
    check.add_argument("--root", type=Path, required=True)
    check.add_argument("--snapshot", type=Path, required=True)
    check.add_argument("--fail-on-new", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "snapshot":
            document = snapshot(args.root, include_source=not args.no_source)
            write_snapshot(document, args.output, args.root)
            print(json.dumps({"snapshot_files": len(document["files"]),
                              "snapshot_bytes_covered": sum(row["bytes"] for row in document["files"].values()),
                              "snapshot_sha256": document["snapshot_sha256"],
                              "baseline_status": document["baseline_status"]}, indent=2))
        else:
            document = json.loads(args.snapshot.read_text(encoding="utf-8"))
            result = verify(args.root, document, fail_on_new=args.fail_on_new)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            if not result["unchanged"]:
                raise SystemExit(2)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"Baseline preservation error: {exc}\n")


if __name__ == "__main__":
    main()
