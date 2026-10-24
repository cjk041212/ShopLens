#!/usr/bin/env python3
"""Install verified ShopLens runtime assets from this Git checkout or an archive.

Every payload is verified before any destination file or directory is created.
Existing identical files are retained; different files are never overwritten.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import os
import re
import shutil
import stat
import sys
import tarfile
import tempfile
import urllib.parse
import urllib.request
import unicodedata
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

DOMAINS = {"baby", "sports", "clothing"}
MODES = {"qwen_frozen", "qwen_semantic", "qwen_preference"}
MANIFEST = "runtime/manifest.json"
REPO_MANIFEST = "assets/runtime/manifest.json"
MAX_REPO_PART_BYTES = 24_000_000
MAX_MEMBERS = 100_000
MAX_MANIFEST_BYTES = 16 * 1024 * 1024
MAX_PAYLOAD_BYTES = 16 * 1024**3
SPACE_RESERVE = 16 * 1024**2
BUFFER = 1024 * 1024


class AssetError(ValueError):
    pass


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(BUFFER), b""):
            digest.update(block)
    return digest.hexdigest()


def _checksum(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", value):
        raise AssetError("A complete 64-character SHA256 checksum is required.")
    return value.lower()


def _relative(value, *, directory=False):
    if not isinstance(value, str) or not value or "\\" in value or ":" in value or any(ord(char) < 32 for char in value):
        raise AssetError("Archive paths must be plain relative POSIX paths.")
    value = value.removesuffix("/") if directory else value
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts) or str(path) != value:
        raise AssetError("Archive traversal or noncanonical path rejected.")
    reserved = {"con", "prn", "aux", "nul", *{f"com{i}" for i in range(1, 10)}, *{f"lpt{i}" for i in range(1, 10)}}
    if any(part.endswith((" ", ".")) or part.split(".", 1)[0].lower() in reserved for part in path.parts):
        raise AssetError("Nonportable or reserved destination path rejected.")
    return value


def _allowed(path):
    parts = PurePosixPath(path).parts
    if len(parts) == 4 and parts[:2] == ("data", "processed") and parts[2] in DOMAINS:
        return parts[3] in {"items.jsonl", "audit.json"}
    if len(parts) == 4 and parts[:2] == ("data", "images") and parts[2] in DOMAINS:
        return PurePosixPath(parts[3]).suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    if len(parts) == 3 and parts[0] == "artifacts" and parts[1] in DOMAINS:
        return parts[2] in {"features.npz", "features.json", "results.json"}
    if len(parts) == 4 and parts[:2] == ("artifacts", "qwen") and parts[2] in DOMAINS:
        return parts[3] == "results.json"
    if len(parts) == 5 and parts[:2] == ("artifacts", "qwen") and parts[2] in DOMAINS and parts[3] == "base":
        return parts[4] in {"features.npz", "features.json"}
    if len(parts) >= 6 and parts[:2] == ("artifacts", "qwen") and parts[2] in DOMAINS and parts[3] in MODES and parts[4] == "seed42":
        if len(parts) == 6:
            return parts[5] in {"complete.json", "candidate.npz", "serving.npz", "results.json", "training.json"}
        if len(parts) == 7 and parts[5] == "models":
            return parts[6] in {parts[3] + ".json", parts[3] + ".pt"}
        if len(parts) == 7 and parts[5] == "adapter" and parts[3] != "qwen_frozen":
            return parts[6] in {"README.md", "adapter_config.json", "adapter_model.safetensors"}
    return False


@dataclass
class Entry:
    name: str
    size: int
    directory: bool
    native: object


class Archive:
    def __init__(self, path):
        self.source = zipfile.ZipFile(path) if zipfile.is_zipfile(path) else tarfile.open(path, "r:*")
        self.entries = []
        names = set()
        try:
            members = self.source.infolist() if isinstance(self.source, zipfile.ZipFile) else iter(self.source)
            for index, member in enumerate(members):
                if index >= MAX_MEMBERS:
                    raise AssetError("Archive has too many members.")
                if isinstance(member, zipfile.ZipInfo):
                    mode = member.external_attr >> 16
                    directory = member.is_dir()
                    if member.flag_bits & 1 or stat.S_ISLNK(mode) or stat.S_IFMT(mode) not in (0, stat.S_IFDIR, stat.S_IFREG):
                        raise AssetError("Encrypted, symbolic-link or special ZIP member rejected.")
                    name, size = member.filename, member.file_size
                else:
                    directory = member.isdir()
                    if not (directory or member.isreg()) or member.issparse():
                        raise AssetError("Link, sparse or special TAR member rejected.")
                    name, size = member.name, member.size
                name = _relative(name, directory=directory)
                portable_key = unicodedata.normalize("NFC", name).casefold()
                if portable_key in names:
                    raise AssetError("Duplicate archive path rejected.")
                names.add(portable_key)
                if size < 0 or size > MAX_PAYLOAD_BYTES:
                    raise AssetError("Archive member size exceeds the safety limit.")
                self.entries.append(Entry(name, size, directory, member))
        except Exception:
            self.source.close()
            raise

    def open(self, entry):
        return self.source.open(entry.native) if isinstance(self.source, zipfile.ZipFile) else self.source.extractfile(entry.native)

    def close(self):
        self.source.close()


def _manifest_files(manifest):
    rows = manifest.get("files")
    if not isinstance(rows, list) or not rows or len(rows) >= MAX_MEMBERS:
        raise AssetError("Manifest must list a bounded, nonempty payload.")
    files, names = {}, set()
    total = 0
    for row in rows:
        if not isinstance(row, dict):
            raise AssetError("Manifest file records must be objects.")
        path = _relative(row.get("path"))
        if not _allowed(path):
            raise AssetError(f"Path is outside the public runtime asset allowlist: {path}")
        portable_key = unicodedata.normalize("NFC", path).casefold()
        if portable_key in names:
            raise AssetError("Duplicate manifest payload path rejected.")
        names.add(portable_key)
        size = row.get("bytes")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise AssetError("Manifest payload sizes must be nonnegative integers.")
        checksum = _checksum(row.get("sha256"))
        total += size
        if total > MAX_PAYLOAD_BYTES:
            raise AssetError("Bundle uncompressed size exceeds the safety limit.")
        files[path] = {"bytes": size, "sha256": checksum}
    return files, total


def _read_manifest(archive):
    candidates = [entry for entry in archive.entries if entry.name == MANIFEST and not entry.directory]
    if len(candidates) != 1 or candidates[0].size > MAX_MANIFEST_BYTES:
        raise AssetError("Bundle must contain one bounded runtime/manifest.json.")
    with archive.open(candidates[0]) as stream:
        manifest = json.loads(stream.read(MAX_MANIFEST_BYTES + 1).decode("utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1 or manifest.get("archive_root") != "runtime":
        raise AssetError("Unsupported runtime asset manifest schema.")
    files, total = _manifest_files(manifest)
    expected = {"runtime/" + path for path in files} | {MANIFEST}
    actual = {entry.name for entry in archive.entries if not entry.directory}
    parents = {str(parent) for path in expected for parent in PurePosixPath(path).parents if str(parent) != "."}
    if actual != expected or any(entry.directory and entry.name not in parents for entry in archive.entries):
        raise AssetError("Archive payload differs from its exact checksum manifest.")
    for entry in archive.entries:
        if not entry.directory and entry.name != MANIFEST and entry.size != files[entry.name[len("runtime/"):]]["bytes"]:
            raise AssetError("Archive member size does not match the manifest.")
    return manifest, files, total


def _destination(root, relative):
    destination = root.joinpath(*PurePosixPath(relative).parts)
    current = root
    if root.exists() and not root.is_dir():
        raise AssetError("Installation root is not a directory.")
    for part in PurePosixPath(relative).parts:
        current = current / part
        if current.is_symlink():
            raise AssetError(f"Existing destination symlink rejected: {relative}")
        if current != destination and current.exists() and not current.is_dir():
            raise AssetError(f"Destination parent is not a directory: {relative}")
    if destination.exists() and not destination.is_file():
        raise AssetError(f"Destination is not a regular file: {relative}")
    return destination


def _free_space(path):
    path = Path(path)
    while not path.exists():
        path = path.parent
    return shutil.disk_usage(path).free


def _reserve(size):
    return max(SPACE_RESERVE, min(256 * 1024**2, size // 20))


def _filesystem_block_size(path):
    path = Path(path)
    while not path.exists():
        path = path.parent
    try:
        filesystem = os.statvfs(path)
        return max(4096, filesystem.f_frsize or filesystem.f_bsize)
    except (AttributeError, OSError):
        return 4096


def _verify_stream(stream, size, expected, output=None):
    digest, count = hashlib.sha256(), 0
    for block in iter(lambda: stream.read(BUFFER), b""):
        count += len(block)
        if count > size:
            raise AssetError("Payload exceeds its declared size.")
        digest.update(block)
        if output is not None:
            output.write(block)
    if count != size or digest.hexdigest() != expected:
        raise AssetError("Payload content SHA256 or size mismatch.")


def _repo_source(root, relative):
    path = _destination(root, relative)
    if not path.is_file():
        raise AssetError(f"Missing repository asset: {relative}")
    return path


def _model_source_allowed(path):
    parts = PurePosixPath(path).parts
    if len(parts) not in (6, 7) or parts[:2] != ("models", "qwen") or parts[2] not in DOMAINS or parts[3] not in MODES or parts[4] != "seed42":
        return False
    if len(parts) == 6:
        return parts[5] in {"head.pt", "head.json"}
    return parts[3] != "qwen_frozen" and parts[5] == "adapter" and parts[6] in {"README.md", "adapter_config.json", "adapter_model.safetensors"}


def _read_repo_manifest(repo):
    path = _repo_source(repo, REPO_MANIFEST)
    if path.stat().st_size > MAX_MANIFEST_BYTES:
        raise AssetError("Repository manifest exceeds its safety limit.")
    with path.open("rb") as stream:
        encoded = stream.read(MAX_MANIFEST_BYTES + 1)
    if len(encoded) > MAX_MANIFEST_BYTES:
        raise AssetError("Repository manifest exceeds its safety limit.")
    manifest = json.loads(encoded.decode("utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 2:
        raise AssetError("Unsupported repository asset manifest schema.")
    files, total = _manifest_files(manifest)
    rows = manifest.get("parts")
    if not isinstance(rows, list) or len(rows) >= MAX_MEMBERS:
        raise AssetError("Repository manifest must list bounded runtime parts.")
    parts, packed_bytes = [], 0
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or row.get("path") != f"assets/runtime/chunks/part-{index:03d}.bin":
            raise AssetError("Repository parts must have consecutive canonical paths.")
        size = row.get("bytes")
        if isinstance(size, bool) or not isinstance(size, int) or not 0 < size <= MAX_REPO_PART_BYTES:
            raise AssetError("Repository part size exceeds its safety limit.")
        source = _repo_source(repo, row["path"])
        if source.stat().st_size != size:
            raise AssetError("Repository part size mismatch.")
        parts.append({"path": source, "offset": packed_bytes, "bytes": size, "sha256": _checksum(row.get("sha256"))})
        packed_bytes += size
        if packed_bytes > MAX_PAYLOAD_BYTES:
            raise AssetError("Repository parts exceed the payload safety limit.")
    ranges, direct_paths = [], set()
    for row in manifest["files"]:
        record = files[row["path"]]
        source = row.get("source")
        if not isinstance(source, dict):
            raise AssetError("Repository payload must identify its source.")
        if set(source) == {"path"}:
            relative = _relative(source["path"])
            if not _model_source_allowed(relative) or relative in direct_paths:
                raise AssetError("Repository model source is disallowed or duplicated.")
            direct_paths.add(relative)
            record["source"] = _repo_source(repo, relative)
            if record["source"].stat().st_size != record["bytes"]:
                raise AssetError("Repository model size mismatch.")
        elif set(source) == {"offset"}:
            offset = source["offset"]
            if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0 or offset + record["bytes"] > packed_bytes:
                raise AssetError("Repository payload range is outside the runtime parts.")
            record["offset"] = offset
            ranges.append((offset, record["bytes"]))
        else:
            raise AssetError("Repository payload must have exactly one source path or offset.")
    end = 0
    for offset, size in sorted(ranges):
        if offset != end:
            raise AssetError("Repository payload ranges overlap or leave unlisted bytes.")
        end += size
    if end != packed_bytes:
        raise AssetError("Repository payload ranges do not cover the runtime parts.")
    return manifest, files, total, parts


class _PartStream:
    """A bounded payload view, including payloads crossing Git chunk boundaries."""

    def __init__(self, parts, offset, size):
        self.parts, self.position, self.remaining = parts, offset, size
        self.index = bisect.bisect_right([part["offset"] for part in parts], offset) - 1
        self.stream = None

    def read(self, size):
        wanted, blocks = min(size, self.remaining), []
        while wanted:
            part = self.parts[self.index]
            if self.stream is None:
                self.stream = part["path"].open("rb")
                self.stream.seek(self.position - part["offset"])
            block = self.stream.read(min(wanted, part["offset"] + part["bytes"] - self.position))
            if not block:
                break
            blocks.append(block)
            self.position += len(block)
            self.remaining -= len(block)
            wanted -= len(block)
            if self.position == part["offset"] + part["bytes"]:
                self.close()
                self.index += 1
        return b"".join(blocks)

    def close(self):
        if self.stream is not None:
            self.stream.close()
            self.stream = None


@contextmanager
def _open_repo_payload(record, parts):
    stream = record["source"].open("rb") if "source" in record else _PartStream(parts, record["offset"], record["bytes"])
    try:
        yield stream
    finally:
        stream.close()


def _install_payload(manifest, files, total, root, open_payload, *, dry_run=False, verify_sources=None):
    skipped, new_bytes, allocation = set(), 0, 0
    block = _filesystem_block_size(root)
    for relative, record in files.items():
        target = _destination(root, relative)
        if target.exists():
            if target.stat().st_size != record["bytes"] or sha256(target) != record["sha256"]:
                raise AssetError(f"Refusing to overwrite an existing different file: {relative}")
            skipped.add(relative)
        else:
            new_bytes += record["bytes"]
            # Small photos need whole filesystem blocks plus per-file metadata.
            allocation += ((record["bytes"] + block - 1) // block) * block + 4096
    free = _free_space(root)
    needed = allocation + _reserve(new_bytes)
    if allocation and free < needed:
        raise AssetError(f"Insufficient disk space: need {needed:,} free bytes ({new_bytes:,} payload, {allocation - new_bytes:,} estimated filesystem overhead, plus reserve), have {free:,}. Nothing was installed.")
    if verify_sources is not None:
        verify_sources()
    for record in files.values():
        with open_payload(record) as stream:
            _verify_stream(stream, record["bytes"], record["sha256"])
    installed = 0
    if not dry_run:
        for relative, record in files.items():
            if relative in skipped:
                continue
            target = _destination(root, relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            _destination(root, relative)
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(target, flags, 0o644)
            try:
                with os.fdopen(descriptor, "wb") as output, open_payload(record) as stream:
                    _verify_stream(stream, record["bytes"], record["sha256"], output)
            except Exception:
                target.unlink(missing_ok=True)
                raise
            installed += 1
    return {"ok": True, "dry_run": dry_run, "bundle_name": manifest.get("bundle_name"),
            "payload_files": len(files), "payload_bytes": total,
            "new_payload_bytes": new_bytes, "minimum_free_bytes": needed, "available_free_bytes": free,
            "estimated_filesystem_overhead_bytes": allocation - new_bytes, "filesystem_block_bytes": block,
            "identical_files_retained": len(skipped), "files_installed": installed,
            "root": str(root), "models_included": "Selected seed42 adapters/heads; base encoder weights are prepared separately."}


def install(archive_path, expected_sha256, root, *, dry_run=False):
    archive_path = Path(archive_path).expanduser().resolve()
    checksum = _checksum(expected_sha256)
    if sha256(archive_path) != checksum:
        raise AssetError("Archive SHA256 mismatch; nothing was installed.")
    root = Path(root).expanduser().resolve()
    archive = Archive(archive_path)
    try:
        manifest, files, total = _read_manifest(archive)
        for entry in archive.entries:
            if not entry.directory and entry.name != MANIFEST:
                files[entry.name[len("runtime/"):]]["entry"] = entry
        report = _install_payload(manifest, files, total, root, lambda record: archive.open(record["entry"]), dry_run=dry_run)
        report["archive_sha256"] = checksum
        return report
    finally:
        archive.close()


def install_repo(repo, root, *, dry_run=False):
    repo = Path(repo).expanduser().absolute()
    if repo.is_symlink() or not repo.is_dir():
        raise AssetError("Repository root must be an existing directory, not a symlink.")
    repo = repo.resolve()
    manifest, files, total, parts = _read_repo_manifest(repo)

    def verify_parts():
        for part in parts:
            with part["path"].open("rb") as stream:
                _verify_stream(stream, part["bytes"], part["sha256"])

    report = _install_payload(manifest, files, total, Path(root).expanduser().resolve(),
                              lambda record: _open_repo_payload(record, parts),
                              dry_run=dry_run, verify_sources=verify_parts)
    report["repository"] = str(repo)
    return report


@contextmanager
def _download(url):
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password:
        raise AssetError("Asset URL must be public HTTP(S), without embedded credentials.")
    with tempfile.TemporaryDirectory(prefix="shoplens-assets-") as directory:
        destination = Path(directory) / "runtime-assets.download"
        request = urllib.request.Request(url, headers={"User-Agent": "ShopLens-asset-installer/1"})
        with urllib.request.urlopen(request, timeout=60) as response:
            if urllib.parse.urlsplit(response.geturl()).scheme not in ("http", "https"):
                raise AssetError("Non-HTTP(S) asset redirect rejected.")
            announced = response.headers.get("Content-Length")
            announced = int(announced) if announced and announced.isdigit() else None
            free = _free_space(destination.parent)
            if announced is not None and (announced > MAX_PAYLOAD_BYTES or free < announced + _reserve(announced)):
                raise AssetError("Insufficient temporary disk space or oversized asset download; nothing was installed.")
            count = 0
            with destination.open("xb") as output:
                for block in iter(lambda: response.read(BUFFER), b""):
                    count += len(block)
                    if count > MAX_PAYLOAD_BYTES or count + _reserve(count) > free:
                        raise AssetError("Asset download exceeds the available temporary space or safety limit.")
                    output.write(block)
            if announced is not None and count != announced:
                raise AssetError("Asset download length does not match Content-Length.")
        yield destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--archive", type=Path, help="Local runtime tar.gz or ZIP bundle.")
    source.add_argument("--url", help="Published public HTTP(S) runtime bundle URL.")
    source.add_argument("--repo", type=Path, help="Git checkout containing models/ and assets/runtime/manifest.json.")
    parser.add_argument("--sha256", help="Full archive SHA256; required with --archive or --url.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--dry-run", action="store_true", help="Verify manifest, files, destination conflicts and disk space without installing.")
    args = parser.parse_args()
    try:
        if args.repo is not None:
            if args.sha256 is not None:
                raise AssetError("--sha256 is for archive sources, not --repo.")
            result = install_repo(args.repo, args.root, dry_run=args.dry_run)
        elif args.archive is not None:
            result = install(args.archive, args.sha256, args.root, dry_run=args.dry_run)
        else:
            _checksum(args.sha256)
            with _download(args.url) as downloaded:
                result = install(downloaded, args.sha256, args.root, dry_run=args.dry_run)
        print(json.dumps(result, indent=2))
    except (AssetError, OSError, ValueError, EOFError, tarfile.TarError, zipfile.BadZipFile) as exc:
        print(f"Asset installation failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
