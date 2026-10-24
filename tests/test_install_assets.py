"""Tiny public-asset fixtures; never reads or extracts real coursework assets."""

import hashlib
import importlib.util
import io
import json
import stat
import subprocess
import sys
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/install_assets.py"
spec = importlib.util.spec_from_file_location("shoplens_asset_installer_fixture", SCRIPT)
installer = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = installer
spec.loader.exec_module(installer)


class AssetInstallerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="shoplens-assets-tiny-fixture-")
        self.addCleanup(self.temporary.cleanup)
        self.work = Path(self.temporary.name)
        self.root = self.work / "target"
        self.payload = {"data/processed/baby/items.jsonl": b'{"item_id":"DIAGNOSTIC-PRODUCT"}\n'}

    def bundle(self, *, payload=None, manifest_changes=None, extra=None, archive_type="tar"):
        payload = self.payload if payload is None else payload
        manifest = {"schema_version": 1, "bundle_name": "DIAGNOSTIC ONLY", "archive_root": "runtime", "files": [
            {"path": path, "bytes": len(value), "sha256": hashlib.sha256(value).hexdigest()} for path, value in payload.items()]}
        if manifest_changes:
            manifest_changes(manifest)
        members = [("runtime/manifest.json", json.dumps(manifest).encode()), *[("runtime/" + path, value) for path, value in payload.items()]]
        members += extra or []
        archive = self.work / ("fixture.zip" if archive_type == "zip" else "fixture.tar.gz")
        if archive_type == "zip":
            with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as output:
                for name, value in members:
                    output.writestr(name, value)
        else:
            with tarfile.open(archive, "w:gz") as output:
                for name, value in members:
                    if isinstance(name, tarfile.TarInfo):
                        output.addfile(name)
                    else:
                        info = tarfile.TarInfo(name)
                        info.size = len(value)
                        output.addfile(info, io.BytesIO(value))
        return archive, installer.sha256(archive)

    def assert_no_install(self):
        self.assertFalse(self.root.exists())

    def test_tar_and_zip_install_verified_bytes_and_idempotently_skip_identical_files(self):
        for archive_type in ("tar", "zip"):
            with self.subTest(archive_type=archive_type):
                archive, sha = self.bundle(archive_type=archive_type)
                report = installer.install(archive, sha, self.root)
                self.assertEqual(report["files_installed"], 1 if archive_type == "tar" else 0)
                self.assertEqual((self.root / next(iter(self.payload))).read_bytes(), next(iter(self.payload.values())))
                again = installer.install(archive, sha, self.root)
                self.assertEqual(again["files_installed"], 0)
                self.assertEqual(again["identical_files_retained"], 1)

    def test_dry_run_fully_checks_payload_without_creating_destination(self):
        archive, sha = self.bundle()
        report = installer.install(archive, sha, self.root, dry_run=True)
        self.assertTrue(report["dry_run"])
        self.assertEqual(report["payload_files"], 1)
        self.assertEqual(report["files_installed"], 0)
        self.assertGreater(report["minimum_free_bytes"], report["new_payload_bytes"])
        self.assert_no_install()

    def test_wrong_archive_or_payload_checksum_never_creates_destination(self):
        archive, sha = self.bundle()
        with self.assertRaisesRegex(installer.AssetError, "Archive SHA256"):
            installer.install(archive, "0" * 64, self.root)
        archive, sha = self.bundle(manifest_changes=lambda document: document["files"][0].update(sha256="0" * 64))
        with self.assertRaisesRegex(installer.AssetError, "Payload content SHA256"):
            installer.install(archive, sha, self.root)
        self.assert_no_install()

    def test_existing_conflict_is_detected_before_any_new_payload_is_written(self):
        payload = {**self.payload, "data/processed/baby/audit.json": b'{"fixture_only":true}'}
        conflicting = self.root / "data/processed/baby/audit.json"
        conflicting.parent.mkdir(parents=True)
        conflicting.write_bytes(b"KEEP ORIGINAL")
        archive, sha = self.bundle(payload=payload)
        with self.assertRaisesRegex(installer.AssetError, "existing different file"):
            installer.install(archive, sha, self.root)
        self.assertEqual(conflicting.read_bytes(), b"KEEP ORIGINAL")
        self.assertFalse((self.root / next(iter(self.payload))).exists())

    def test_disk_space_preflight_fails_before_creating_any_target(self):
        archive, sha = self.bundle()
        with patch.object(installer, "_free_space", return_value=0), self.assertRaisesRegex(installer.AssetError, "Insufficient disk space"):
            installer.install(archive, sha, self.root)
        self.assert_no_install()

    def test_many_tiny_files_include_block_and_metadata_space_before_any_writes(self):
        payload = {f"data/images/baby/fixture-{index}.jpg": b"x" for index in range(100)}
        archive, sha = self.bundle(payload=payload)
        # Logical bytes and the old reserve fit; actual small-file allocation does not.
        free = installer.SPACE_RESERVE + 100_000
        with patch.object(installer, "_free_space", return_value=free), patch.object(installer, "_filesystem_block_size", return_value=4096):
            with self.assertRaisesRegex(installer.AssetError, "filesystem overhead"):
                installer.install(archive, sha, self.root)
        self.assert_no_install()

    def test_manifest_traversal_and_private_dataset_paths_are_not_installable(self):
        paths = ("../escape.txt", "/absolute", "data/../escape.txt", "data\\escape.txt", "C:/escape.txt",
                 "data/processed/baby/reviews_train.jsonl", "data/processed/baby/events.jsonl",
                 "artifacts/qwen/baby/qwen_frozen/seed43/training.json",
                 "data/images/baby/CON.jpg",
                 "data/studio_hf_cache/model.safetensors", "src/shoplens/studio.py")
        for path in paths:
            with self.subTest(path=path):
                archive, sha = self.bundle(payload={path: b"DIAGNOSTIC ONLY"})
                with self.assertRaises(installer.AssetError):
                    installer.install(archive, sha, self.root)
                self.assert_no_install()

    def test_tar_symlink_hardlink_and_fifo_are_rejected_without_writes(self):
        for kind in (tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.FIFOTYPE):
            member = tarfile.TarInfo("runtime/data/processed/baby/unsafe")
            member.type, member.linkname = kind, "../../outside"
            archive, sha = self.bundle(extra=[(member, b"")])
            with self.subTest(kind=kind), self.assertRaisesRegex(installer.AssetError, "special TAR"):
                installer.install(archive, sha, self.root)
            self.assert_no_install()

    def test_zip_symlink_and_traversal_are_rejected_without_writes(self):
        link = zipfile.ZipInfo("runtime/data/images/baby/symlink.jpg")
        link.create_system = 3
        link.external_attr = (stat.S_IFLNK | 0o777) << 16
        for extra in ([(link, b"../../outside")], [("../../escape.txt", b"DIAGNOSTIC")]):
            archive, sha = self.bundle(archive_type="zip", extra=extra)
            with self.subTest(extra=str(extra[0][0])), self.assertRaises(installer.AssetError):
                installer.install(archive, sha, self.root)
            self.assert_no_install()

    def test_duplicate_members_and_unlisted_members_are_rejected(self):
        for extra in ([("runtime/" + next(iter(self.payload)), b"duplicate")], [("runtime/data/processed/baby/audit.json", b"unlisted")]):
            archive, sha = self.bundle(extra=extra)
            with self.subTest(extra=extra), self.assertRaises(installer.AssetError):
                installer.install(archive, sha, self.root)
            self.assert_no_install()

    def test_case_aliases_cannot_install_two_members_to_one_portable_path(self):
        archive, sha = self.bundle(payload={"data/images/baby/fixture.jpg": b"first", "data/images/baby/FIXTURE.jpg": b"second"})
        with self.assertRaisesRegex(installer.AssetError, "Duplicate archive"):
            installer.install(archive, sha, self.root)
        self.assert_no_install()

    def test_existing_parent_symlink_cannot_escape_installation_root(self):
        outside = self.work / "outside"
        outside.mkdir()
        self.root.mkdir()
        (self.root / "data").symlink_to(outside, target_is_directory=True)
        archive, sha = self.bundle()
        with self.assertRaisesRegex(installer.AssetError, "destination symlink"):
            installer.install(archive, sha, self.root)
        self.assertEqual(list(outside.iterdir()), [])

    def test_aggregate_metrics_and_exact_selected_adapter_files_are_allowed(self):
        paths = ("artifacts/baby/results.json", "artifacts/qwen/sports/results.json",
                 "artifacts/qwen/baby/qwen_frozen/seed42/training.json",
                 "artifacts/qwen/clothing/qwen_semantic/seed42/adapter/README.md",
                 "artifacts/qwen/clothing/qwen_semantic/seed42/adapter/adapter_model.safetensors")
        archive, sha = self.bundle(payload={path: b"DIAGNOSTIC ONLY" for path in paths})
        report = installer.install(archive, sha, self.root)
        self.assertEqual(report["files_installed"], 5)

    def test_public_url_validation_occurs_before_any_network_request(self):
        for url in ("file:///tmp/forbidden", "https://account@example.invalid/runtime.tar.gz", "ftp://example.invalid/runtime"):
            with self.subTest(url=url), patch.object(installer.urllib.request, "urlopen") as opener:
                with self.assertRaises(installer.AssetError), installer._download(url):
                    pass
                opener.assert_not_called()

    def test_cli_imports_and_help_work_without_site_packages_or_pip(self):
        completed = subprocess.run([sys.executable, "-S", str(SCRIPT), "--help"], capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--sha256", completed.stdout)


class RepositoryAssetInstallerTests(unittest.TestCase):
    """Exercise a tiny checkout with split data and a directly tracked model."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="shoplens-repo-assets-fixture-")
        self.addCleanup(temporary.cleanup)
        self.work = Path(temporary.name)
        self.repo, self.root = self.work / "repo", self.work / "target"
        self.payload = {
            "data/processed/baby/items.jsonl": b'{"item_id":"FIXTURE"}\n',
            "data/processed/baby/audit.json": b'{"fixture_only":true}',
            "artifacts/qwen/baby/qwen_frozen/seed42/models/qwen_frozen.pt": b"DIAGNOSTIC MODEL BYTES",
        }
        self.model_source = "models/qwen/baby/qwen_frozen/seed42/head.pt"
        self.manifest_path = self.repo / installer.REPO_MANIFEST
        packed, rows = bytearray(), []
        for path, content in self.payload.items():
            source = {"offset": len(packed)}
            if path.endswith(".pt"):
                source = {"path": self.model_source}
                model = self.repo / self.model_source
                model.parent.mkdir(parents=True)
                model.write_bytes(content)
            else:
                packed.extend(content)
            rows.append({"path": path, "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest(), "source": source})
        parts = []
        for index, offset in enumerate(range(0, len(packed), 13)):
            content = packed[offset:offset + 13]
            path = f"assets/runtime/chunks/part-{index:03d}.bin"
            target = self.repo / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
            parts.append({"path": path, "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()})
        self.manifest = {"schema_version": 2, "files": rows, "parts": parts}
        self.write_manifest()

    def write_manifest(self):
        self.manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")

    def assert_rejected_without_writes(self, message=None):
        with self.assertRaisesRegex(installer.AssetError, message or "."):
            installer.install_repo(self.repo, self.root)
        self.assertFalse(self.root.exists())

    def test_installs_split_payload_and_direct_model_and_resumes_partial_checkout(self):
        existing = self.root / next(iter(self.payload))
        existing.parent.mkdir(parents=True)
        existing.write_bytes(self.payload[next(iter(self.payload))])
        source_hashes = {path: installer.sha256(path) for path in self.repo.rglob("*") if path.is_file()}
        with patch.object(installer.urllib.request, "urlopen") as network:
            report = installer.install_repo(self.repo, self.root)
            network.assert_not_called()
        self.assertEqual(report["files_installed"], 2)
        self.assertEqual(report["identical_files_retained"], 1)
        for path, content in self.payload.items():
            self.assertEqual((self.root / path).read_bytes(), content)
        self.assertEqual(source_hashes, {path: installer.sha256(path) for path in source_hashes})
        again = installer.install_repo(self.repo, self.root)
        self.assertEqual(again["files_installed"], 0)
        self.assertEqual(again["identical_files_retained"], 3)

    def test_dry_run_verifies_every_source_without_creating_target(self):
        report = installer.install_repo(self.repo, self.root, dry_run=True)
        self.assertTrue(report["dry_run"])
        self.assertEqual(report["files_installed"], 0)
        self.assertEqual(report["payload_files"], 3)
        self.assertFalse(self.root.exists())

    def test_corrupt_chunk_is_rejected_before_creating_target(self):
        part = self.repo / self.manifest["parts"][0]["path"]
        part.write_bytes(b"!" + part.read_bytes()[1:])
        self.assert_rejected_without_writes("Payload content SHA256")

    def test_individual_payload_hash_is_checked_even_when_part_hash_matches(self):
        part = self.repo / self.manifest["parts"][0]["path"]
        part.write_bytes(b"!" + part.read_bytes()[1:])
        self.manifest["parts"][0]["sha256"] = installer.sha256(part)
        self.write_manifest()
        self.assert_rejected_without_writes("Payload content SHA256")

    def test_corrupt_direct_model_is_rejected_before_any_data_install(self):
        model = self.repo / self.model_source
        model.write_bytes(b"!" + model.read_bytes()[1:])
        self.assert_rejected_without_writes("Payload content SHA256")

    def test_invalid_ranges_are_rejected_before_any_target_creation(self):
        original = self.manifest["files"][1]["source"]["offset"]
        for offset in (-1, True, 10**9, 0, original + 1):
            with self.subTest(offset=offset):
                self.manifest["files"][1]["source"]["offset"] = offset
                self.write_manifest()
                self.assert_rejected_without_writes("range")

    def test_unlisted_chunk_tail_and_missing_parts_are_rejected(self):
        self.manifest["files"].pop(1)
        self.write_manifest()
        self.assert_rejected_without_writes("cover")
        (self.repo / self.manifest["parts"][0]["path"]).unlink()
        self.assert_rejected_without_writes("Missing repository asset")

    def test_source_paths_cannot_read_private_or_traversing_files(self):
        for source in ("../private", "/absolute", "data/private.json", "models/qwen/baby/qwen_frozen/seed43/head.pt"):
            with self.subTest(source=source):
                self.manifest["files"][-1]["source"] = {"path": source}
                self.write_manifest()
                self.assert_rejected_without_writes()

    def test_source_file_and_parent_symlinks_are_rejected(self):
        model = self.repo / self.model_source
        outside = self.work / "outside-model"
        model.rename(outside)
        model.symlink_to(outside)
        self.assert_rejected_without_writes("symlink")
        model.unlink()
        outside.rename(model)
        chunks = self.repo / "assets/runtime/chunks"
        outside_chunks = self.work / "outside-chunks"
        chunks.rename(outside_chunks)
        chunks.symlink_to(outside_chunks, target_is_directory=True)
        self.assert_rejected_without_writes("symlink")

    def test_existing_conflict_and_low_disk_fail_before_any_new_payload(self):
        conflicting = self.root / "data/processed/baby/audit.json"
        conflicting.parent.mkdir(parents=True)
        conflicting.write_bytes(b"KEEP ORIGINAL")
        with self.assertRaisesRegex(installer.AssetError, "existing different file"):
            installer.install_repo(self.repo, self.root)
        self.assertEqual(conflicting.read_bytes(), b"KEEP ORIGINAL")
        self.assertFalse((self.root / next(iter(self.payload))).exists())
        conflicting.unlink()
        with patch.object(installer, "_free_space", return_value=0), self.assertRaisesRegex(installer.AssetError, "Insufficient disk space"):
            installer.install_repo(self.repo, self.root)
        self.assertEqual(list(self.root.rglob("*.*")), [])

    def test_malformed_source_and_oversize_part_metadata_are_rejected(self):
        self.manifest["files"][0]["source"] = {"offset": 0, "path": self.model_source}
        self.write_manifest()
        self.assert_rejected_without_writes("exactly one source")
        self.manifest["parts"][0]["bytes"] = installer.MAX_REPO_PART_BYTES + 1
        self.write_manifest()
        self.assert_rejected_without_writes("part size")

    def test_repo_cli_uses_only_stdlib_and_needs_no_archive_checksum(self):
        completed = subprocess.run([sys.executable, "-S", str(SCRIPT), "--repo", str(self.repo), "--root", str(self.root), "--dry-run"], capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(json.loads(completed.stdout)["payload_files"], 3)
        self.assertFalse(self.root.exists())


if __name__ == "__main__":
    unittest.main()
