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


if __name__ == "__main__":
    unittest.main()
