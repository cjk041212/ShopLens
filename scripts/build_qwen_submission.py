#!/usr/bin/env python3
"""Build the independent baseline+Qwen code/evidence submission; default dry-run.

Example (does not write anything):
  python scripts/build_qwen_submission.py --root /path/to/NLP-Coursework
Explicit build after reviewing the manifest:
  python scripts/build_qwen_submission.py --root /path/to/NLP-Coursework --write

Only archive copies are sanitized. Source files are never changed. No network,
model loading, subprocesses, Git operations, job submission, or school upload.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import zipfile
from zoneinfo import ZoneInfo

DOMAINS = ("baby", "sports", "clothing")
MODELS = ("popularity", "mf", "tfidf", "text", "multimodal", "adapted")
QWEN_MODES = ("qwen_frozen", "qwen_semantic", "qwen_preference")
SEEDS = (42, 43, 44)
MAX_FILE_BYTES = 16 * 1024 * 1024
MAX_TOTAL_BYTES = 100 * 1024 * 1024
PRIVATE_LOCATION = re.compile(r"/(?:Users|home)/[^/\s\"'<>]+(?:/[^\s\"'<>)]*)?|/(?:private/)?var/folders/[^\s\"'<>)]*")
KEY_BLOCK = re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----")
SECRET_TOKEN = re.compile(r"\b(?:AKIA[0-9A-Z]{16}|gh[pousr]_[A-Za-z0-9]{30,}|hf_[A-Za-z0-9]{25,}|sk-[A-Za-z0-9]{24,})\b")
CREDENTIAL_KEY = re.compile(r"^(?:SSH_.*|password|passwd|private_key|ssh_private_key|secret|api_key|access_token|auth_token|refresh_token|ssh_user|ssh_host|cluster_host|username|user|account|job_owner|account_name|pbs_o_logname|logname|pbs_project|pbs_account|slurm_account|project_account)$", re.I)
PRIVATE_IP = re.compile(r"(?<![\w.])(?:10\.\d{1,3}\.\d{1,3}\.\d{1,3}|192\.168\.\d{1,3}\.\d{1,3}|172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3})(?![\w.])")
URL_CREDENTIAL = re.compile(r"https?://[^/\s:@]+:[^/\s@]+@")
AUTH_ASSIGNMENT = re.compile(r"^\s*(?:export\s+)?(?:[A-Z_]*(?:PASSWORD|PASSWD|SECRET|API_KEY|ACCESS_TOKEN|AUTH_TOKEN|PRIVATE_KEY)|password|passwd|secret|api_key|access_token|auth_token|private_key)\s*[:=]\s*['\"]?[^\s'\"#<${]", re.M)


class PackagingError(ValueError):
    pass


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def baseline_inventory_matches(root: Path, snapshot: dict) -> bool:
    """Recheck the original snapshot, so a stale unchanged proof cannot pass."""
    files = snapshot.get("files")
    content = {key: value for key, value in snapshot.items() if key != "snapshot_sha256"}
    canonical = json.dumps(content, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    if (snapshot.get("schema_version") != 1 or snapshot.get("hash_algorithm") != "sha256"
            or snapshot.get("snapshot_sha256") != digest(canonical) or not isinstance(files, dict) or not files):
        return False
    root = root.resolve()
    try:
        for relative, expected in files.items():
            name = Path(relative)
            path = root / name
            if (name.is_absolute() or ".." in name.parts or "\\" in relative
                    or not path.resolve().is_relative_to(root) or not isinstance(expected, dict)
                    or any(part.is_symlink() for part in (path, *path.parents) if part.is_relative_to(root))
                    or not path.is_file()):
                return False
            before = path.stat()
            actual = file_digest(path)
            after = path.stat()
            identity = lambda stat: (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
            if (identity(before) != identity(after) or actual != expected.get("sha256")
                    or after.st_size != expected.get("bytes")):
                return False
    except (OSError, TypeError, ValueError):
        return False
    return True


def json_bytes(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def private_path_to_relative(match: re.Match) -> str:
    value = match.group(0)
    for marker in ("/NLP-Coursework/", "/nlp_coursework_shoplens/"):
        if marker in value:
            return value.split(marker, 1)[1]
    if value.endswith(("/NLP-Coursework", "/nlp_coursework_shoplens")):
        return "."
    return "[PRIVATE_PATH]/" + value.rstrip("/").rsplit("/", 1)[-1]


def clean_string(value: str) -> str:
    value = PRIVATE_LOCATION.sub(private_path_to_relative, value)
    return PRIVATE_IP.sub("[PRIVATE_ADDRESS]", value)


def clean_json(value, changes: set[str]):
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if CREDENTIAL_KEY.fullmatch(str(key)) and item not in (None, "", False):
                result[key] = "<REDACTED_CREDENTIAL>"
                changes.add("credential-field-redacted")
            else:
                result[key] = clean_json(item, changes)
        return result
    if isinstance(value, list):
        return [clean_json(item, changes) for item in value]
    if isinstance(value, str):
        cleaned = clean_string(value)
        if cleaned != value:
            changes.add("private-paths-made-relative-or-redacted")
        return cleaned
    return value


def assert_safe(text: str, filename: str) -> None:
    # Report the file, never the matched credential or private location.
    checks = ((KEY_BLOCK, "private-key block"), (SECRET_TOKEN, "credential token"),
              (URL_CREDENTIAL, "URL credential"), (PRIVATE_LOCATION, "private absolute path"),
              (PRIVATE_IP, "private network address"), (AUTH_ASSIGNMENT, "authentication assignment"))
    for pattern, reason in checks:
        if pattern.search(text):
            raise PackagingError(f"Refusing {filename}: {reason} remains after sanitization")


def portable_presentation_source(text: str) -> str:
    text = re.sub(r"^const (dataRoot|root) = .*;$",
                  lambda match: f"const {match.group(1)} = path.resolve(option('--root', process.env.SHOPLENS_ROOT || workspaceDir));",
                  text, flags=re.M)
    text = re.sub(r"^const (runtimeBase|runtime) = .*;$",
                  lambda match: f"const {match.group(1)} = process.env.SHOPLENS_RUNTIME_BASE || '';", text, flags=re.M)
    text = re.sub(r"^const skill = .*;$",
                  "const skill = process.env.SHOPLENS_PRESENTATION_SKILL || '';", text, flags=re.M)
    return text


def sanitize_pptx(payload: bytes, filename: str, changes: set[str]) -> bytes:
    source = zipfile.ZipFile(io.BytesIO(payload))
    output = io.BytesIO()
    with source, zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as target:
        for member in source.infolist():
            if member.filename.startswith("/") or ".." in Path(member.filename).parts:
                raise PackagingError(f"Unsafe PPTX member in {filename}")
            data = source.read(member)
            if member.filename.endswith((".xml", ".rels")):
                text = data.decode("utf-8")
                cleaned = clean_string(text)
                if text != cleaned:
                    changes.add("pptx-private-paths-redacted")
                assert_safe(cleaned, filename + ":" + member.filename)
                data = cleaned.encode("utf-8")
            target.writestr(member, data)
    return output.getvalue()


def sanitize(payload: bytes, relative: Path) -> tuple[bytes, list[str]]:
    changes: set[str] = set()
    filename = relative.as_posix()
    if relative.suffix == ".pptx":
        data = sanitize_pptx(payload, filename, changes)
        if data != payload:
            changes.add("pptx-container-repacked")
        return data, sorted(changes)
    if relative.suffix.lower() in {".png", ".jpg", ".jpeg"}:
        # Browser screenshot tools may emit JPEG bytes under a .png source name.
        # Preserve the evidence bytes and filename; do not re-encode user media.
        if not payload.startswith((b"\x89PNG\r\n\x1a\n", b"\xff\xd8\xff")):
            raise PackagingError(f"Unrecognized raster screenshot: {filename}")
        if relative.suffix.lower() in {".jpg", ".jpeg"} and not payload.startswith(b"\xff\xd8\xff"):
            raise PackagingError(f"Invalid JPEG screenshot signature: {filename}")
        assert_safe(payload.decode("latin-1"), filename)
        return payload, (["jpeg-screenshot-source-name-preserved"]
                         if relative.suffix.lower() == ".png" and payload.startswith(b"\xff\xd8\xff") else [])
    text = payload.decode("utf-8")
    if relative.suffix == ".json":
        original_value = json.loads(text)
        if json_bytes(original_value) != payload:
            changes.add("json-format-normalized")
        data = json_bytes(clean_json(original_value, changes))
        assert_safe(data.decode("utf-8"), filename)
        return data, sorted(changes)
    if relative.name in {"build_presentation.mjs", "build_qwen_presentation.mjs"}:
        text = portable_presentation_source(text)
        changes.add("optional-presentation-generator-runtime-configurable")
    if relative.suffix == ".pbs":
        cleaned = re.sub(r"^#PBS\s+-P\s+\S+.*$",
                         "# Supply your scheduler project at submission: qsub -P PROJECT_CODE <script>",
                         text, flags=re.M)
        if text != cleaned:
            changes.add("scheduler-project-identifier-removed")
        text = cleaned
    cleaned = clean_string(text)
    if text != cleaned:
        changes.add("private-paths-made-relative-or-redacted")
    assert_safe(cleaned, filename)
    return cleaned.encode("utf-8"), sorted(changes)


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def public_status(root: Path) -> dict:
    expected = {(domain, model, seed) for domain in DOMAINS for model in MODELS for seed in SEEDS}
    found = Counter()
    invalid = []
    for domain in DOMAINS:
        path = root / "artifacts" / domain / "results.json"
        if not path.exists():
            continue
        for row in load_json(path).get("results", []):
            if row.get("split") != "test":
                continue
            key = (domain, row.get("model"), row.get("seed"))
            if (row.get("data_audit", {}).get("synthetic") or
                    row.get("features_metadata", {}).get("synthetic") or
                    row.get("run_protocol", {}).get("evaluation_subset")):
                invalid.append([domain, str(row.get("model")), row.get("seed")])
                continue
            found[key] += 1
    missing = sorted(expected - found.keys())
    duplicates = [[*key, count] for key, count in found.items() if count > 1]
    unexpected = sorted(found.keys() - expected)
    budgets = []
    for domain in DOMAINS:
        for seed in SEEDS:
            if (root / "artifacts/budget_check" / domain / f"seed{seed}" / "budget_check.json").exists():
                budgets.append([domain, seed])
    baseline = {"expected_primary_records": len(expected), "unique_primary_records": len(expected & found.keys()),
            "primary_complete": not missing and not duplicates and not invalid and not unexpected,
            "missing_primary_records": missing, "duplicate_primary_records": duplicates,
            "unexpected_primary_records": [list(key) for key in unexpected],
            "rejected_synthetic_or_diagnostic_records": invalid,
            "expected_budget_checks": 9, "completed_budget_checks": len(budgets),
            "budget_checks": budgets}
    old_proof = load_json(root / "deliverables/verification.json") if (root / "deliverables/verification.json").is_file() else {}
    baseline_verified = (old_proof.get("status") == "passed" and not old_proof.get("errors")
                         and old_proof.get("formal_records") == 54 and old_proof.get("budget_records") == 9)
    proof_path = root / "experiments/qwen/verification.json"
    proof = load_json(proof_path) if proof_path.is_file() else {}
    summary_path = root / "deliverables/results_qwen_summary.json"
    summary = load_json(summary_path) if summary_path.is_file() else {}
    qwen_expected = {(domain, mode, seed) for domain in DOMAINS for mode in QWEN_MODES for seed in SEEDS}
    qwen_found = Counter()
    qwen_errors = []
    recovery_path = root / "experiments/qwen/recovery_note.json"
    recovery = load_json(recovery_path) if recovery_path.is_file() else None
    if (proof.get("execution_recovery") != recovery or
            (recovery_path.is_file() and proof.get("execution_recovery_source_sha256") != file_digest(recovery_path))):
        qwen_errors.append("Recovery note changed or was not included in the current Qwen verification")
    numerical_present = isinstance(recovery, dict) and "numerical_recovery" in recovery
    numerical = recovery.get("numerical_recovery") if numerical_present else None
    if numerical_present and (not isinstance(numerical, dict) or numerical.get("status") != "complete"
            or any(numerical.get(key) != "passed" for key in
                   ("production_same_state_regression", "production_prefix100_regression"))):
        qwen_errors.append("Numerical recovery or production regression gates are not complete")
    for domain in DOMAINS:
        path = root / "artifacts/qwen" / domain / "results.json"
        if not path.is_file():
            continue
        for row in load_json(path).get("results", []):
            if row.get("split") != "test":
                continue
            key = domain, row.get("model"), row.get("seed")
            qwen_found[key] += 1
            if (row.get("data_audit", {}).get("synthetic") or row.get("features_metadata", {}).get("synthetic")
                    or row.get("run_protocol", {}).get("evaluation_subset") or row.get("diagnostic")):
                qwen_errors.append(f"Synthetic/diagnostic Qwen result: {domain}")
            if key not in qwen_expected:
                qwen_errors.append("Unexpected Qwen result identity")
                continue
            mode, seed = key[1:]
            unit = root / "artifacts/qwen" / domain / mode / f"seed{seed}"
            paths = (unit / "training.json", unit / "models" / f"{mode}.json", unit / "complete.json", unit / "results.json")
            if not all(path.is_file() for path in paths):
                qwen_errors.append(f"Missing selected JSON evidence: {domain}/{mode}/seed{seed}")
                continue
            training, metadata, complete, result = map(load_json, paths)
            if (result.get("results") != [row] or complete.get("results_sha256") != digest(paths[3].read_bytes())
                    or complete.get("selected_epoch") not in (1, 2)
                    or training.get("diagnostic") or metadata.get("diagnostic") or complete.get("diagnostic")):
                qwen_errors.append(f"Invalid completed-unit evidence: {domain}/{mode}/seed{seed}")
            evidence = proof.get("unit_evidence", {}).get(f"{domain}/{mode}/seed{seed}", {})
            expected_hashes = evidence.get("source_evidence_sha256", {})
            if any(expected_hashes.get(path.relative_to(unit).as_posix()) != file_digest(path) for path in paths):
                qwen_errors.append(f"Selected JSON evidence changed since verification: {domain}/{mode}/seed{seed}")
            if numerical_present and domain == "clothing" and mode != "qwen_frozen":
                numerics = training.get("training_numerics") or {}
                if (not isinstance(numerics, dict) or numerics.get("candidate_training_sdpa_backend") != "math"
                        or numerics.get("nonfinite_gradient_guard") is not True
                        or numerics.get("cached_encoding_sdpa_backend") != "automatic"):
                    qwen_errors.append(f"Missing Clothing MATH/guard/automatic-cache policy: {mode}/seed{seed}")
    qwen_verified = (proof.get("complete") is True and proof.get("expected_records") == 27
                     and proof.get("summarized_records") == 27 and not proof.get("errors") and not proof.get("missing_units")
                     and proof.get("local_cache_bytes_verified") is True
                     and len(proof.get("unit_evidence", {})) == 27
                     and summary.get("complete") is True and summary.get("raw_record_count") == 27
                     and summary.get("verification") == proof)
    preservation_path = root / "experiments/qwen/baseline_verification.json"
    before_path = root / "experiments/qwen/baseline_before.json"
    preservation = load_json(preservation_path) if preservation_path.is_file() else {}
    before = load_json(before_path) if before_path.is_file() else {}
    preservation_verified = (preservation.get("unchanged") is True and bool(before.get("files"))
                             and preservation.get("snapshot_sha256") == before.get("snapshot_sha256")
                             and preservation.get("matched_files") == len(before.get("files", {}))
                             and baseline_inventory_matches(root, before))
    required = ("README_QWEN.md", "requirements-qwen.txt", "experiments/qwen/protocol.md",
                "experiments/qwen/model_source.json", "deliverables/ShopLens_Qwen_report.md",
                "deliverables/ShopLens_Qwen_report_en.md", "deliverables/results_qwen_summary.csv",
                "deliverables/ShopLens_Qwen_presentation.pptx", "deliverables/ShopLens_Qwen_human_narration.md")
    missing_documents = [name for name in required if not (root / name).is_file()]
    environments = [path for path in (root / "experiments/qwen").rglob("*environment*.json") if path.is_file()]
    matrix_complete = (set(qwen_found) == qwen_expected and all(count == 1 for count in qwen_found.values()))
    all_complete = (baseline["primary_complete"] and len(budgets) == 9 and baseline_verified
                    and matrix_complete and qwen_verified and preservation_verified and not qwen_errors
                    and not missing_documents and bool(environments))
    return {**baseline, "baseline_verification_passed": baseline_verified,
            "qwen_expected_records": 27, "qwen_unique_records": len(qwen_expected & qwen_found.keys()),
            "qwen_matrix_complete": matrix_complete, "qwen_verification_passed": qwen_verified,
            "qwen_evidence_errors": qwen_errors, "baseline_preservation_verified": preservation_verified,
            "missing_required_documents": missing_documents, "qwen_environment_evidence_present": bool(environments),
            "all_final_requirements_met": all_complete}


def candidates(root: Path) -> list[Path]:
    result: set[Path] = set()
    fixed = ("README.md", "README_QWEN.md", "requirements.txt", "requirements-qwen.txt", "requirements-lock-macos.txt", "pyproject.toml",
             "app.py", "启动演示.command", ".streamlit/config.toml")
    result.update(root / name for name in fixed if (root / name).is_file())
    rules = {"src": {".py"}, "tests": {".py"},
             "scripts": {".py", ".sh", ".pbs", ".mjs"},
             "configs": {".json", ".yaml", ".yml", ".toml", ".ini", ".md"},
             "deliverables": {".md", ".csv", ".json", ".pptx"}}
    for directory, suffixes in rules.items():
        for path in (root / directory).rglob("*"):
            if (path.is_file() and path.suffix in suffixes and
                    not path.name.endswith(".zip.manifest.json") and
                    not any(part.startswith(".") or part in {"__pycache__", "node_modules"}
                            or part.endswith(".egg-info") for part in path.relative_to(root).parts)):
                result.add(path)
    for path in (root / "data/raw").glob("*.source.json"):
        result.add(path)
    for domain in DOMAINS:
        for filename in ("audit.json",):
            path = root / "data/processed" / domain / filename
            if path.exists():
                result.add(path)
        for filename in ("results.json", "features.json", "text_adaptation/adaptation.json"):
            path = root / "artifacts" / domain / filename
            if path.exists():
                result.add(path)
        for path in (root / "experiments/runs" / domain).glob("seed*/*.json"):
            result.add(path)
        for path in (root / "artifacts/budget_check" / domain).glob("seed*/*.json"):
            result.add(path)
        for directory in (root / "artifacts" / domain / "models", root / "experiments/runs" / domain,
                          root / "artifacts/budget_check" / domain, root / "artifacts/qwen" / domain):
            result.update(path for path in directory.rglob("*.json") if path.is_file()
                          and not any(part.startswith(".") or part in {"logs", "__pycache__"}
                                      for part in path.relative_to(root).parts))
    for directory, suffixes in ((root / "experiments/qwen", {".json", ".md", ".csv"}),
                                (root / "deliverables/assets", {".png", ".jpg", ".jpeg", ".svg"})):
        result.update(path for path in directory.rglob("*") if path.is_file() and path.suffix.lower() in suffixes
                      and not any(part.startswith(".") or part in {"logs", "__pycache__"}
                                  for part in path.relative_to(root).parts))
    result.update(path for path in (root / "experiments/qwen/diagnostics").glob("*/original_source/*.py")
                  if path.is_file() and not any(part.startswith(".") for part in path.relative_to(root).parts))
    for filename in ("local_benchmark.json",):
        path = root / "artifacts" / filename
        if path.exists():
            result.add(path)
    for filename in ("baseline_budget.json", "complete.json"):
        path = root / "experiments" / filename
        if path.exists():
            result.add(path)
    return sorted(result, key=lambda path: path.relative_to(root).as_posix())


def build_entries(root: Path) -> tuple[dict[str, bytes], dict]:
    root = root.resolve()
    status = public_status(root)
    entries = {}
    records = []
    for path in candidates(root):
        relative = path.relative_to(root)
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise PackagingError(f"Refusing a symlink or outside-root file: {relative}")
        if path.stat().st_size > MAX_FILE_BYTES:
            raise PackagingError(f"Selected file exceeds size limit: {relative}")
        original = path.read_bytes()
        packaged, changes = sanitize(original, relative)
        name = "ShopLens/" + relative.as_posix()
        entries[name] = packaged
        records.append({"path": relative.as_posix(), "source_sha256": digest(original),
                        "archive_sha256": digest(packaged), "archive_bytes": len(packaged),
                        "transformations": changes})
    # Keep a small image coverage audit rather than thousands of per-item URLs/errors.
    for domain in DOMAINS:
        source = root / "data/processed" / domain / "images_audit.json"
        if source.exists():
            original = source.read_bytes()
            audit = json.loads(original)
            summary = {key: audit[key] for key in ("seed", "catalog_items", "sample_items", "all_items_checked",
                       "download_requested", "counts", "note") if key in audit}
            summary.update({"source_images_audit_sha256": digest(original),
                            "packaging_note": "Per-item records omitted to keep the code package small."})
            relative = f"data/processed/{domain}/images_audit.summary.json"
            payload = json_bytes(summary)
            entries["ShopLens/" + relative] = payload
            records.append({"path": relative, "source_sha256": digest(original),
                            "archive_sha256": digest(payload), "archive_bytes": len(payload),
                            "transformations": ["image-audit-coverage-summary"]})
    reproduction = {
        "domains": list(DOMAINS), "methods": list(MODELS), "recommender_seeds": list(SEEDS),
        "primary_epochs": 5, "adaptation_epochs": 2, "adaptation_seed": 42,
        "adaptation_max_pairs": 30000, "recommender_batch_size": 256,
        "data_preparation": {"positive_rating_at_least": 4, "min_user_positive_events": 3,
                             "require_text": True, "require_decodable_images": True},
        "note": "Declared protocol; per-run complete.json and result metadata are the measured evidence.",
    }
    entries["ShopLens/configs/submission_protocol.json"] = json_bytes(reproduction)
    entries["ShopLens/configs/submission_qwen_protocol.json"] = json_bytes({
        "model_id": "Qwen/Qwen3-Embedding-0.6B", "revision": "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
        "domains": list(DOMAINS), "objectives": list(QWEN_MODES), "full_pipeline_seeds": list(SEEDS),
        "warmup": {"max_epochs": 20, "min_epochs": 5, "patience": 3, "selection": "full-catalog validation NDCG@10"},
        "adaptation": {"epochs": 2, "max_cases": 30000, "selected_epoch": "1or2 by downstream validation NDCG@10"},
        "encoding": {"max_length": 256, "padding_side": "left", "pooling": "last valid token", "normalization": "L2"},
        "histories": "frozen base-Qwen text + original fixed CLIP; never adapted candidate embeddings",
        "note": "Declared configuration; per-unit training/model/result JSON and verification are the measured evidence.",
    })
    note = """# Independent baseline + Qwen submission package

This is a code, report, and result-evidence package. Raw reviews, raw product photo files,
processed interaction/catalog files, model weights, feature matrices, virtual
environments, cache files, account configuration, logs, and original course
materials are deliberately absent. The UI initially shows its documented empty
state; reconstruct data and train artifacts before a model demo.
Small UI screenshots and illustrations inside the presentations are retained as
presentation evidence; the source photo dataset is not included.

Run the Python setup/tests and data commands in README.md, then run
`scripts/run_experiments.py` for the retained legacy comparison. Then follow
README_QWEN.md and install requirements-qwen.txt for the independent Qwen upgrade.
Use an allocated GPU and the declared domain/seed/objective configuration. The public-data download needs network access. Original files
are identified by SHA256 in the audit and source sidecars. Image availability
may change; validate the resulting catalog hashes before claiming exact replay.
The macOS lock file records the local environment; it is not a CUDA lock file.
The final package requires54 baseline records,9 baseline budget checks,27 Qwen
records, the full local-cache Qwen verification, and baseline preservation proof.
These checks concern experimental evidence; they do not replace human course forms.

Source code is preserved except for explicitly listed archive-only privacy and
portability transformations. Audit/result absolute private paths become project
relative paths. Their original source hashes and packaged hashes are both in
SUBMISSION_MANIFEST.json; hashes inside scientific records still identify the
original experiment inputs, not the redacted archive view.

The optional Node presentation generator requires externally supplied
`@oai/artifact-tool` and presentation helper runtime, outside Python requirements.
Set RUNTIME_NODE_MODULES, RUNTIME_PYTHON and SHOPLENS_PRESENTATION_SKILL explicitly.
Editable PPTX output is included when present; rebuilding it is optional for the
Python recommendation project. Scheduler project identifiers are removed from
PBS templates; choose your own allocated environment and scheduler project.

The final course submission also requires a real-voice video (maximum 10 minutes),
the group's actual contribution reflection with signatures, and the approved AI
disclosure form/records. Narration notes and unsigned drafts are preparation
materials, not completed video or signed forms. The team must also complete the
course peer evaluation separately. No upload or school submission is performed.
"""
    entries["ShopLens/SUBMISSION_README.md"] = note.encode()
    for name in ("ShopLens/configs/submission_protocol.json", "ShopLens/configs/submission_qwen_protocol.json",
                 "ShopLens/SUBMISSION_README.md"):
        records.append({"path": name.removeprefix("ShopLens/"), "source_sha256": None,
                        "archive_sha256": digest(entries[name]), "archive_bytes": len(entries[name]),
                        "transformations": ["generated-packaging-document"]})
    total = sum(map(len, entries.values()))
    if total > MAX_TOTAL_BYTES:
        raise PackagingError("Selected package exceeds total uncompressed size limit")
    manifest = {"schema_version": 1, "created_at": datetime.now(ZoneInfo("Asia/Singapore")).isoformat(),
                "archive_root": "ShopLens", "uncompressed_bytes_before_manifest": total,
                "source_files_unchanged": True, "experimental_status": status,
                "excluded_categories": ["venv", "raw-data", "raw-product-photos", "processed-reviews/catalog/events",
                                        "HF-cache-and-weights", "model-checkpoints", "feature-matrices",
                                        "accounts-and-authentication", "logs", "original-course-materials",
                                        "chat-history-and-invented-AI-records", "existing-submission-ZIPs"],
                "files": sorted(records, key=lambda record: record["path"])}
    entries["ShopLens/SUBMISSION_MANIFEST.json"] = json_bytes(manifest)
    return entries, manifest


def write_archive(entries: dict[str, bytes], output: Path, manifest: dict) -> None:
    if manifest.get("experimental_status", {}).get("all_final_requirements_met") is not True:
        raise PackagingError("Final build requires54 baseline+9 budget+27 Qwen complete verified records, baseline preservation, and final documents")
    if output.exists():
        raise PackagingError("Output archive already exists; choose a new filename")
    sidecar = output.with_suffix(output.suffix + ".manifest.json")
    if sidecar.exists():
        raise PackagingError("Manifest destination already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".part")
    if temporary.exists():
        raise PackagingError("Temporary output already exists; inspect it before retrying")
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for name in sorted(entries):
                info = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                executable = name.endswith((".sh", ".command", ".pbs"))
                info.external_attr = ((0o100755 if executable else 0o100644) << 16)
                archive.writestr(info, entries[name])
        with zipfile.ZipFile(temporary) as archive:
            if archive.testzip() is not None:
                raise PackagingError("Archive integrity verification failed")
            for name, expected in entries.items():
                if archive.read(name) != expected:
                    raise PackagingError("Archive content verification failed")
        temporary.replace(output)
        sidecar.write_bytes(json_bytes({**manifest, "zip_sha256": digest(output.read_bytes()),
                                        "zip_bytes": output.stat().st_size}))
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--write", action="store_true", help="Explicitly build an archive; default is dry-run")
    parser.add_argument("--output", type=Path, help="Defaults to deliverables/ShopLens_Qwen_code_submission.zip")
    args = parser.parse_args(argv)
    try:
        entries, manifest = build_entries(args.root)
        status = manifest["experimental_status"]
        if args.write:
            output = args.output or args.root / "deliverables/ShopLens_Qwen_code_submission.zip"
            write_archive(entries, output, manifest)
        print(json.dumps({"mode": "written" if args.write else "dry-run", "file_count": len(entries),
                          "uncompressed_bytes": sum(map(len, entries.values())),
                          "experimental_status": status,
                          "transformed_files": [{"path": row["path"], "transformations": row["transformations"]}
                                                for row in manifest["files"] if row["transformations"]]},
                         ensure_ascii=False, indent=2))
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        parser.exit(1, f"Packaging error: {exc}\n")


if __name__ == "__main__":
    main()
