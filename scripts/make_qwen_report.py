#!/usr/bin/env python3
"""Audit and summarize independent Qwen runs without changing baseline reports."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
from zoneinfo import ZoneInfo

DOMAINS = ("baby", "sports", "clothing")
MODES = ("qwen_frozen", "qwen_semantic", "qwen_preference")
SEEDS = (42, 43, 44)
METRICS = ("recall@10", "ndcg@10", "recall@20", "ndcg@20")
MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
PROTOCOL_KEYS = ("candidate_set", "history", "max_history", "evaluation", "averaging", "positive_feedback",
                 "repeat_targets", "ties", "eligible_targets", "evaluation_subset", "n_catalog_items", "history_encoder")
EXPECTED_PROTOCOL = {
    "candidate_set": "Full prepared catalog, excluding all strictly earlier positive items",
    "history": "At most max_history distinct most recent positive items with timestamp < target timestamp",
    "max_history": 20,
    "evaluation": "Prequential: earlier split interactions may enter later histories; model weights remain frozen",
    "averaging": "Within-user target mean, then macro-average over users",
    "positive_feedback": "rating >= 4", "repeat_targets": "Skipped when already present in earlier positive history",
    "ties": "Score descending, stable sorted item_id ascending", "evaluation_subset": False,
    "history_encoder": "Frozen base Qwen text and fixed CLIP image features in all three modes",
}


def read_json(path: Path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def tree_hash(directory: Path) -> str:
    value = hashlib.sha256()
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            value.update(str(path.relative_to(directory)).encode())
            value.update(sha256(path).encode())
    return value.hexdigest()


def finite(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def mean_std(values) -> tuple[float, float | None]:
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else None


def metric_summary(rows) -> dict:
    result = {"n_seeds": len(rows), "seeds": sorted(row["seed"] for row in rows)}
    for metric in METRICS:
        values = [row[metric] for row in rows]
        if values:
            result[metric], result[metric + "_std"] = mean_std(values)
    return result


def proof_fields(training: dict) -> dict:
    """Normalize proof names; the stored original JSON remains the evidence."""
    warmup = training.get("warmup") or {}
    hashes = training.get("sample_hashes", training.get("hashes", {}))
    return {
        "warmup_head_sha256": training.get("warmup_head_sha256", warmup.get("head_sha256", warmup.get("state_sha256"))),
        "adaptation_cases_sha256": training.get("adaptation_cases_sha256", hashes.get("adaptation_cases_sha256")),
        "negative_sha256_by_epoch": training.get("negative_sha256_by_epoch", training.get("negative_hashes")),
        "case_order_sha256_by_epoch": training.get("case_order_sha256_by_epoch"),
        "sampling_sha256": training.get("sampling_sha256", hashes.get("sampling_sha256")),
        "gradient_evidence": training.get("gradient_evidence"),
        "live_cache_parity": training.get("live_cache_parity"),
    }


def aggregate(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    groups = defaultdict(list)
    for row in rows:
        groups[row["domain"], row["model"]].append(row)
    summaries = []
    for (domain, mode), values in sorted(groups.items()):
        entry = {"domain": domain, "model": mode, **metric_summary(values)}
        lows = [{"seed": row["seed"], **row["low_interaction"]} for row in values
                if all(finite(row.get("low_interaction", {}).get(metric)) for metric in (*METRICS, "n_users", "n_targets"))]
        if lows:
            entry["low_interaction"] = {**metric_summary(lows), "max_training_positive_users": 5,
                                        "n_users": lows[0]["n_users"], "n_targets": lows[0]["n_targets"]}
        summaries.append(entry)
    paired = []
    for domain in DOMAINS:
        for before, after in (("qwen_frozen", "qwen_semantic"), ("qwen_frozen", "qwen_preference"),
                              ("qwen_semantic", "qwen_preference")):
            left = {row["seed"]: row for row in groups[domain, before]}
            right = {row["seed"]: row for row in groups[domain, after]}
            seeds = sorted(left.keys() & right.keys())
            if not seeds:
                continue
            differences = [right[seed]["ndcg@10"] - left[seed]["ndcg@10"] for seed in seeds]
            mean, std = mean_std(differences)
            paired.append({"domain": domain, "before": before, "after": after, "metric": "ndcg@10",
                           "seeds": seeds, "differences": differences, "mean_difference": mean,
                           "std_difference": std, "wins": sum(value > 1e-12 for value in differences),
                           "ties": sum(abs(value) <= 1e-12 for value in differences),
                           "losses": sum(value < -1e-12 for value in differences)})
    return summaries, paired


def audit_root(root: Path, *, verify_cache_files: bool = False) -> tuple[list[dict], dict]:
    root = Path(root)
    errors, warnings, rows, unit_proofs, unit_evidence = [], [], [], {}, {}

    def error(unit, check, message):
        errors.append({"unit": unit, "check": check, "message": message})

    recovery_path = root / "experiments/qwen/recovery_note.json"
    recovery = read_json(recovery_path)
    numerical_recovery_present = isinstance(recovery, dict) and "numerical_recovery" in recovery
    numerical_recovery = recovery.get("numerical_recovery") if numerical_recovery_present else None
    if numerical_recovery_present and (not isinstance(numerical_recovery, dict)
            or numerical_recovery.get("status") != "complete"
            or any(numerical_recovery.get(name) != "passed" for name in
                   ("production_same_state_regression", "production_prefix100_regression"))):
        error("numerical-recovery", "recovery-regressions",
              "Numerical recovery must be complete with both production regressions passed")

    expected = {(domain, mode, seed) for domain in DOMAINS for mode in MODES for seed in SEEDS}
    found = Counter()
    for domain in DOMAINS:
        documents = read_json(root / "artifacts/qwen" / domain / "results.json", {})
        base_directory = root / "artifacts/qwen" / domain / "base"
        base_metadata = read_json(base_directory / "features.json", {})
        base_checksum = sha256(base_directory / "features.npz") if verify_cache_files and (base_directory / "features.npz").exists() else None
        legacy_metadata = read_json(root / "artifacts" / domain / "features.json", {})
        if verify_cache_files and documents.get("results"):
            legacy_file = root / "artifacts" / domain / "features.npz"
            if (not legacy_file.is_file() or sha256(legacy_file) != legacy_metadata.get("features_sha256")
                    or legacy_metadata.get("features_sha256") != base_metadata.get("identity", {}).get("old_features_sha256")):
                error(domain, "fixed-clip-source", "Old frozen CLIP feature source is missing or changed")
        actual_hashes = {name: sha256(root / "data/processed" / domain / filename)
                         for name, filename in (("items_sha256", "items.jsonl"), ("events_sha256", "events.jsonl"))
                         if (root / "data/processed" / domain / filename).exists()}
        for source in documents.get("results", []):
            if source.get("split") != "test":
                continue
            mode, seed = source.get("model"), source.get("seed")
            key = domain, mode, seed
            unit = f"{domain}/{mode}/seed{seed}"
            if key not in expected:
                error(unit, "identity", "Unexpected model or pipeline seed in formal domain results")
                continue
            found[key] += 1
            if found[key] > 1:
                error(unit, "identity", "Duplicate formal result identity")
                continue
            if (source.get("data_audit", {}).get("synthetic") or
                    source.get("features_metadata", {}).get("synthetic") or
                    source.get("run_protocol", {}).get("evaluation_subset") or source.get("diagnostic")):
                error(unit, "real-full-evaluation", "Synthetic or diagnostic record is excluded")
                continue
            if (source.get("data_audit", {}).get("synthetic") is not False or
                    source.get("features_metadata", {}).get("real_data") is not True):
                error(unit, "real-full-evaluation", "Explicit real-data provenance is missing")
            if not all(finite(source.get(metric)) and 0 <= source[metric] <= 1 for metric in METRICS):
                error(unit, "metrics", "Missing, nonfinite, or out-of-range metric")
                continue
            row = {"domain": domain, **source}
            directory = root / "artifacts/qwen" / domain / mode / f"seed{seed}"
            training = read_json(directory / "training.json", {})
            metadata = read_json(directory / "models" / f"{mode}.json", {})
            complete = read_json(directory / "complete.json", {})
            numerics = training.get("training_numerics")
            if numerical_recovery_present and domain == "clothing" and mode != "qwen_frozen":
                required_numerics = {"candidate_training_sdpa_backend": "math",
                                     "cached_encoding_sdpa_backend": "automatic"}
                if (not isinstance(numerics, dict)
                        or numerics.get("nonfinite_gradient_guard") is not True
                        or any(numerics.get(name) != value for name, value in required_numerics.items())):
                    error(unit, "training-numerics",
                          "Clothing LoRA requires MATH training SDPA, nonfinite guard, and automatic cached encoding")
            if not training or not metadata or not complete:
                error(unit, "completed-unit", "training.json, selected model metadata, or complete.json is missing")
                continue
            if training.get("diagnostic") or metadata.get("diagnostic") or complete.get("diagnostic"):
                error(unit, "real-full-evaluation", "Diagnostic training cannot enter the formal matrix")
                continue
            rows.append(row)
            if any(document.get("model") != mode or document.get("seed") != seed
                   for document in (training, metadata, complete)):
                error(unit, "identity", "Training/model/completion identity differs from the formal result")
            if metadata.get("family") != "qwen_candidate_lora" or metadata.get("embedding_dim") != 128 or metadata.get("max_history") != 20:
                error(unit, "head-geometry", "Selected scorer must use the declared128d asymmetric family and20-item history")
            config = row.get("training_configuration", {})
            source_result = directory / "results.json"
            unit_result = read_json(source_result, {}).get("results", [])
            if (unit_result != [source] or not source_result.is_file()
                    or sha256(source_result) != complete.get("results_sha256")):
                error(unit, "completed-unit", "Domain result differs from the completed unit result or its checksum")
            if (row.get("selected_epoch") not in (1, 2) or metadata.get("selected_epoch") != row.get("selected_epoch")
                    or training.get("selected_epoch") != row.get("selected_epoch") or config.get("epochs") != 2
                    or metadata.get("epochs") != 2 or complete.get("selected_epoch") != row.get("selected_epoch")):
                error(unit, "epoch-selection", "All records must agree on selected adaptation epoch1 or2 and two trained epochs")
            curve = training.get("adaptation_validation_ndcg@10", [])
            if len(curve) != 2 or not all(finite(value) for value in curve):
                error(unit, "epoch-selection", "Need both downstream validation NDCG values")
            elif training.get("selected_epoch") != max(range(2), key=lambda index: curve[index]) + 1:
                error(unit, "epoch-selection", "Selected epoch does not maximize the declared two-epoch validation curve")
            losses = training.get("adaptation_curve", [])
            if (len(losses) != 2 or [record.get("epoch") for record in losses] != [1, 2]
                    or any(not finite(record.get(name)) or record[name] < 0
                           for record in losses for name in ("bpr_loss", "semantic_loss"))
                    or [record.get("validation_ndcg@10") for record in losses] != curve):
                error(unit, "objective-losses", "Raw objective losses and both epoch validation values are required")
            if config.get("objective") != mode or metadata.get("objective") != mode:
                error(unit, "objective", "Declared objective differs from the reported experimental arm")
            fingerprint = training.get("fingerprint")
            if (not fingerprint or fingerprint != metadata.get("config_sha256") or
                    fingerprint != config.get("config_sha256") or fingerprint != complete.get("fingerprint")):
                error(unit, "configuration", "Result, selected model, training, and completion fingerprints must agree")
            for key_name in ("model_id", "revision", "encoding_config", "lora_config"):
                if training.get(key_name) != config.get(key_name) or metadata.get(key_name) != config.get(key_name):
                    error(unit, "configuration", f"Selected source/config differs: {key_name}")
            encoding = config.get("encoding_config") or {}
            expected_encoding = {"max_length": 256, "padding_side": "left", "pooling": "last valid token",
                                 "normalization": "L2", "truncation": True}
            if any(encoding.get(name) != value for name, value in expected_encoding.items()):
                error(unit, "encoding", "Encoding differs from fixed256/left-padding/last-token/L2/truncation protocol")
            if mode != "qwen_frozen":
                lora = config.get("lora_config") or {}
                if (lora.get("r") != 8 or lora.get("lora_alpha") != 16 or lora.get("lora_dropout") != .05
                        or set(lora.get("target_modules", [])) != {"q_proj", "v_proj"}):
                    error(unit, "lora-config", "LoRA geometry differs from the declared experiment")
            sample_count = training.get("sample_metadata", {}).get("sampled_adaptation_cases")
            eligible_count = training.get("sample_metadata", {}).get("eligible_review_cases")
            if (not isinstance(sample_count, int) or not 0 < sample_count <= 30000
                    or sample_count != metadata.get("n_training_examples") or sample_count != config.get("n_training_examples")
                    or config.get("max_examples") != 30000 or not isinstance(eligible_count, int)
                    or sample_count != min(30000, eligible_count)):
                error(unit, "adaptation-sample", "Actual sample counts or declared30000 budget disagree")
            warmup = training.get("warmup") or {}
            warmup_curve = warmup.get("curve", [])
            if (warmup != metadata.get("warmup") or warmup != config.get("warmup")
                    or warmup.get("max_epochs") != 20 or warmup.get("min_epochs") != 5 or warmup.get("patience") != 3
                    or not 5 <= len(warmup_curve) <= 20
                    or warmup.get("selected_epoch") not in range(1, len(warmup_curve) + 1)):
                error(unit, "warmup-budget", "Full-input warmup selection/budget evidence is incomplete or inconsistent")
            if config.get("model_id") != MODEL_ID or config.get("revision") != REVISION:
                error(unit, "pinned-model", "Model ID or revision differs from the declared Qwen source")
            for name, expected_value in EXPECTED_PROTOCOL.items():
                if row.get("run_protocol", {}).get(name) != expected_value:
                    error(unit, "declared-protocol", f"Missing or changed protocol field: {name}")
            if row.get("run_protocol", {}).get("eligible_targets") != row.get("n_targets"):
                error(unit, "population", "Protocol target count differs from measured target count")
            for name in ("items_sha256", "events_sha256"):
                if not row.get(name) or row.get(name) != metadata.get(name):
                    error(unit, "input-hashes", f"Result and model metadata disagree: {name}")
                if name in actual_hashes and row.get(name) != actual_hashes[name]:
                    error(unit, "input-hashes", f"Current processed input changed: {name}")
            for name in ("serving_sha256", "head_sha256"):
                if not row.get(name) or row.get(name) != metadata.get(name):
                    error(unit, "selected-hashes", f"Result and selected model metadata disagree: {name}")
                if complete.get(name) != metadata.get(name):
                    error(unit, "completed-unit", f"Completion marker differs from selected metadata: {name}")
            if not row.get("selected_features_sha256") or row.get("selected_features_sha256") != metadata.get("candidate_sha256"):
                error(unit, "selected-hashes", "Selected candidate hash differs between result and model metadata")
            if (training.get("selected_candidate_sha256") != metadata.get("candidate_sha256") or
                    training.get("selected_serving_sha256") != metadata.get("serving_sha256") or
                    complete.get("candidate_sha256") != metadata.get("candidate_sha256")):
                error(unit, "selected-hashes", "Training/completion proof does not identify the selected candidate and serving cache")
            adapter_hash = metadata.get("adapter_sha256")
            if (row.get("adapter_sha256") != adapter_hash or complete.get("adapter_sha256") != adapter_hash
                    or training.get("selected_adapter_sha256") != adapter_hash
                    or (mode != "qwen_frozen" and (not isinstance(adapter_hash, str) or len(adapter_hash) != 64))
                    or (mode == "qwen_frozen" and adapter_hash is not None)):
                error(unit, "selected-adapter", "Selected adapter proof is missing or inconsistent")
            if not row.get("featurehash") or row.get("featurehash") != metadata.get("base_features_sha256"):
                error(unit, "base-hash", "Frozen history/base feature hash is missing or inconsistent")
            if base_metadata and (row.get("featurehash") != base_metadata.get("features_sha256")
                    or row.get("features_metadata") != base_metadata):
                error(unit, "base-hash", "Result does not match the recorded shared base feature metadata")
            if verify_cache_files and base_checksum != metadata.get("base_features_sha256"):
                error(unit, "local-cache-bytes", "Base feature bytes do not match the frozen-history cache")
            if any(not finite(row.get(name)) or row[name] < 1 for name in ("n_users", "n_targets")):
                error(unit, "population", "Formal test populations must be positive")
            low = row.get("low_interaction", {})
            if low.get("max_training_positive_users") != 5 or not all(finite(low.get(metric)) for metric in METRICS):
                error(unit, "low-interaction", "Missing <=5 training-positive-user cohort metrics")
            proof = proof_fields(training)
            unit_proofs[key] = proof
            for name in ("warmup_head_sha256", "adaptation_cases_sha256", "sampling_sha256"):
                if not isinstance(proof[name], str) or len(proof[name]) != 64:
                    error(unit, "shared-cases-and-warmup", f"Missing proof: {name}")
            for name in ("negative_sha256_by_epoch", "case_order_sha256_by_epoch"):
                values = proof[name]
                if (not isinstance(values, list) or len(values) != 2
                        or any(not isinstance(value, str) or len(value) != 64 for value in values)):
                    error(unit, "shared-negative-and-order", f"Need two epoch hashes: {name}")
            gradients = proof["gradient_evidence"] or {}
            head_norm = gradients.get("head_gradient_max_norm")
            total_lora = gradients.get("total_lora_gradient_max_norm")
            bpr_lora = gradients.get("bpr_to_lora_max_norm")
            if gradients.get("base_parameters_frozen") is not True:
                error(unit, "gradients", "Base-parameter freezing evidence is missing")
            if not finite(head_norm) or head_norm <= 0:
                error(unit, "gradients", "No nonzero head-gradient evidence")
            parameters, probes = gradients.get("trainable_lora_parameters"), gradients.get("gradient_probes")
            if mode != "qwen_frozen" and (not finite(total_lora) or total_lora <= 0
                    or not finite(parameters) or parameters <= 0 or not finite(probes) or not 1 <= probes <= 10):
                error(unit, "gradients", "Trainable objective has no nonzero LoRA-gradient evidence")
            if mode == "qwen_frozen" and (total_lora != 0 or bpr_lora != 0 or gradients.get("trainable_lora_parameters") != 0):
                error(unit, "gradients", "Frozen objective unexpectedly reports LoRA gradients")
            if mode == "qwen_semantic" and bpr_lora != 0:
                error(unit, "gradient-route", "Semantic BPR must be detached from LoRA")
            if mode == "qwen_preference" and (not finite(bpr_lora) or bpr_lora <= 0):
                error(unit, "gradient-route", "Preference BPR must have a nonzero LoRA gradient")
            parity = proof["live_cache_parity"] or {}
            candidate_error, history_error = parity.get("candidate_max_abs_error"), parity.get("history_score_max_abs_error")
            candidate_tolerance, history_tolerance = parity.get("candidate_tolerance"), parity.get("history_tolerance")
            if (not all(finite(value) for value in (candidate_error, history_error, candidate_tolerance, history_tolerance))
                    or not 0 < candidate_tolerance <= .020000001 or not 0 < history_tolerance <= .000010000001
                    or not 0 <= candidate_error <= candidate_tolerance or not 0 <= history_error <= history_tolerance):
                error(unit, "live-cache-parity", "Live/cache encoding parity has not passed")
            unit_evidence[unit] = {"selected_epoch": training.get("selected_epoch"),
                                  "sample_metadata": training.get("sample_metadata"),
                                  "warmup": training.get("warmup"),
                                  "warmup_selected_epoch": training.get("warmup", {}).get("selected_epoch"),
                                  "adaptation_validation_ndcg@10": curve,
                                  "adaptation_curve": losses,
                                  "shared_training_hashes": {key: proof[key] for key in
                                      ("warmup_head_sha256", "adaptation_cases_sha256", "sampling_sha256",
                                       "negative_sha256_by_epoch", "case_order_sha256_by_epoch")},
                                  "gradient_evidence": gradients, "live_cache_parity": parity,
                                  "training_numerics": numerics,
                                  "source_evidence_sha256": {name: sha256(directory / name) for name in
                                      ("training.json", f"models/{mode}.json", "complete.json", "results.json")
                                      if (directory / name).is_file()},
                                  "candidate_sha256": metadata.get("candidate_sha256"),
                                  "adapter_sha256": metadata.get("adapter_sha256")}
            for name, filename in (("head_sha256", f"models/{mode}.pt"),
                                   ("serving_sha256", "serving.npz"), ("candidate_sha256", "candidate.npz")):
                path = directory / filename
                if verify_cache_files:
                    if not path.is_file() or sha256(path) != metadata.get(name):
                        error(unit, "local-cache-bytes", f"Local cache missing or hash mismatch: {filename}")
            if verify_cache_files and mode != "qwen_frozen":
                if not (directory / "adapter").is_dir() or tree_hash(directory / "adapter") != adapter_hash:
                    error(unit, "local-cache-bytes", "Selected adapter directory is missing or changed")
    missing = sorted(expected - found.keys())
    for domain in DOMAINS:
        domain_rows = [row for row in rows if row["domain"] == domain]
        if domain_rows:
            reference = domain_rows[0]
            for row in domain_rows[1:]:
                unit = f"{domain}/{row['model']}/seed{row['seed']}"
                for name in ("items_sha256", "events_sha256", "featurehash", "n_users", "n_targets"):
                    if row.get(name) != reference.get(name):
                        error(unit, "same-domain-comparison", f"Compared data/base/population differs: {name}")
                for name in PROTOCOL_KEYS:
                    if row.get("run_protocol", {}).get(name) != reference.get("run_protocol", {}).get(name):
                        error(unit, "same-domain-protocol", f"Ranking protocol differs: {name}")
                for name in ("n_users", "n_targets", "max_training_positive_users"):
                    if row.get("low_interaction", {}).get(name) != reference.get("low_interaction", {}).get(name):
                        error(unit, "same-low-cohort", f"Low-interaction cohort differs: {name}")
        for seed in SEEDS:
            proofs = [(mode, unit_proofs[domain, mode, seed]) for mode in MODES if (domain, mode, seed) in unit_proofs]
            if len(proofs) == 3:
                reference = proofs[0][1]
                for mode, proof in proofs[1:]:
                    for name in ("warmup_head_sha256", "adaptation_cases_sha256", "sampling_sha256",
                                 "negative_sha256_by_epoch", "case_order_sha256_by_epoch"):
                        if proof[name] != reference[name]:
                            error(f"{domain}/{mode}/seed{seed}", "paired-training-inputs", f"Objectives do not share {name}")
        frozen = [row for row in domain_rows if row["model"] == "qwen_frozen"]
        if len({row.get("selected_features_sha256") for row in frozen}) > 1:
            error(domain, "frozen-candidates", "Frozen base candidate features differ across pipeline seeds")
    if not verify_cache_files:
        warnings.append("Cache byte hashes were not re-read locally; result/model metadata and training proof declarations were audited.")
    baseline = read_json(root / "experiments/qwen/baseline_verification.json", {})
    if baseline and baseline.get("unchanged") is not True:
        error("baseline", "preservation", "Baseline verification does not report unchanged bytes")
    if not baseline:
        warnings.append("No baseline_verification.json supplied; this report does not independently claim baseline byte preservation.")
    verification = {"schema_version": 1, "generated_at": datetime.now(ZoneInfo("Asia/Singapore")).isoformat(),
                    "complete": not missing and not errors and len(rows) == 27,
                    "expected_records": 27, "observed_identities": len(found), "summarized_records": len(rows),
                    "missing_units": [list(key) for key in missing], "errors": errors, "warnings": warnings,
                    "local_cache_bytes_verification_requested": verify_cache_files,
                    "local_cache_bytes_verified": verify_cache_files and not errors and len(rows) == 27 and not missing,
                    "baseline_preservation": baseline or None,
                    "live_cache_numerical_probe_scope": "First min(4,N) live-Qwen candidates and one validation history at selected export; not full-catalog live-Qwen re-encoding equality",
                    "unit_evidence": unit_evidence}
    if recovery_path.is_file():
        verification["execution_recovery"] = recovery
        verification["execution_recovery_source_sha256"] = sha256(recovery_path)
    return rows, verification


def format_value(row, name):
    value, std = row.get(name), row.get(name + "_std")
    if value is None:
        return "—"
    return f"{value:.6f} ± {std:.6f}" if std is not None else f"{value:.6f}"


def render_report(summary: dict, *, english: bool = False) -> str:
    verification = summary["verification"]
    complete = verification["complete"]
    heading = "# ShopLens Qwen objective comparison" if english else "# ShopLens Qwen 训练目标比较"
    status = ("COMPLETE — all 27 records passed the declared evidence audit." if complete else
              "INCOMPLETE — the full 27-record evidence audit has not passed; no completed-study conclusion is available.")
    if not english:
        status = ("完成：27 条结果通过本报告的声明与证据审计。" if complete else
                  "未完成：27 条结果及训练证据尚未全部通过核对，不能作出完成实验的结论。")
    lines = [heading, "", status, "", f"{verification['summarized_records']} / 27 records; {len(verification['errors'])} audit errors.", ""]
    if english:
        lines += ["The three benchmarks are domains of the same Amazon2014 source; they are not independently sourced datasets. Course acceptance of this three-domain interpretation remains to be confirmed.", "",
                  f"Model: `{MODEL_ID}`, pinned revision `{REVISION}`. The declared encoding uses at most256 tokens, left padding, last-valid-token pooling and L2 normalization, with identical training/export truncation. Candidate descriptions receive no query instruction. Trainable candidates use LoRA r8, alpha16, dropout0.05 on q_proj/v_proj; the recommendation head has128 dimensions.", "",
                  "The primary contrast is an asymmetric two-tower experiment. Histories always use frozen base-Qwen text plus the original fixed CLIP features. Only the candidate text tower changes: frozen Qwen, semantic LoRA, or preference LoRA. Candidate features must not replace history features.", "",
                  "Each complete pipeline seed is repeated independently (42/43/44). Within a domain/seed, all objectives share the selected full-input warmup head, the same sampled behavioral cases and order, and the same negative arrays. Adaptation uses up to30,000 eligible TRAIN cases with exactly matched positive reviews, over two epochs. Select epoch1 or2 by full-catalog downstream validation NDCG@10; epoch0 and test scores cannot select a checkpoint.", "",
                  "Semantic LoRA aligns frozen base-Qwen review anchors with live candidate descriptions using multi-positive InfoNCE; its BPR text input is detached so BPR updates the head. Preference LoRA receives BPR gradients directly. Target reviews never enter recommendation-history queries. All groups retain the same old temporal, exclusion, candidate and user-macro evaluation protocol.", ""]
    else:
        lines += ["三个基准来自同一 Amazon2014 数据源的三个领域，并非三个独立来源。课程是否接受三个领域的口径仍需确认。", "",
                  f"模型：`{MODEL_ID}`，固定 revision `{REVISION}`。编码固定最多256 token、左侧 padding、最后有效 token 池化与 L2 归一化，训练和导出采用相同截断方式。候选商品描述不加 query instruction；可训练候选塔在 q_proj/v_proj 使用 LoRA r8、alpha16、dropout0.05，推荐头为128维。", "",
                  "这是非对称双塔实验：历史塔始终使用冻结的 base-Qwen 文本与原有固定 CLIP 图片特征；候选文本塔分别为冻结 Qwen、语义 LoRA、偏好 LoRA。不能把适配后的候选向量用于历史塔。", "",
                  "每个 seed42/43/44 独立重跑完整流水线；同域同 seed 的三种目标共享选中的全量 warmup 头、相同行为案例及顺序、相同负样本数组。适配从匹配正面评论的 TRAIN 行为中抽取最多30,000例，训练两个 epoch，以完整候选目录的下游验证 NDCG@10 在 epoch1/2 中选优，epoch0 和测试分数不参与选择。", "",
                  "语义 LoRA 用冻结 base-Qwen 评论锚点与实时商品描述做多正例 InfoNCE，BPR 输入文本向量 detach 后只更新推荐头；偏好 LoRA 直接接收 BPR 梯度。目标评论不进入用户历史查询。三组沿用原有时序、候选排除与用户宏平均评价协议。", ""]
    lines += ["## Metrics" if english else "## 实测指标", "",
              "Mean ± sample SD across pipeline seeds; incomplete rows are provisional." if english else
              "完整流水线种子的均值 ± 样本标准差；未完成状态下的表格仅供检查已有记录。", "",
              "| Domain | Objective | Seeds | Recall@10 | NDCG@10 | Recall@20 | NDCG@20 |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for row in summary["results"]:
        lines.append(f"| {row['domain']} | {row['model']} | {row['n_seeds']} | " +
                     " | ".join(format_value(row, metric) for metric in METRICS) + " |")
    lines += ["", "## Paired changes" if english else "## 同 seed 配对变化", "",
              "| Domain | After − before | Seeds | ΔNDCG@10 mean ± SD | Wins / ties / losses |",
              "|---|---|---|---:|---:|"]
    for row in summary["paired_comparisons"]:
        value = f"{row['mean_difference']:+.6f}"
        if row["std_difference"] is not None:
            value += f" ± {row['std_difference']:.6f}"
        lines.append(f"| {row['domain']} | {row['after']} − {row['before']} | {','.join(map(str,row['seeds']))} | {value} | {row['wins']} / {row['ties']} / {row['losses']} |")
    if complete:
        lines.append("")
        for row in summary["paired_comparisons"]:
            direction = "higher" if row["mean_difference"] > 1e-12 else "lower" if row["mean_difference"] < -1e-12 else "unchanged"
            if english:
                lines.append(f"- {row['domain']}, {row['after']} versus {row['before']}: mean NDCG@10 is {direction}; per-seed wins/ties/losses are {row['wins']}/{row['ties']}/{row['losses']}. This describes the declared runs only.")
            else:
                word = {"higher": "增加", "lower": "下降", "unchanged": "持平"}[direction]
                lines.append(f"- {row['domain']}，{row['after']} 相比 {row['before']}：NDCG@10 配对均值{word}，逐 seed 改善/持平/下降为 {row['wins']}/{row['ties']}/{row['losses']}；仅描述本次声明的运行。")
    lines += ["", "## Low-interaction targets" if english else "## 低交互商品", "",
              "Targets with <=5 distinct positive TRAIN users, including zero. This is not a separately constructed strict cold-start test." if english else
              "训练期不同正反馈用户数 ≤5 的目标商品，包含零交互商品；这不是专门构造的严格冷启动测试。", "",
              "| Domain | Objective | Seeds | Users / targets | Recall@10 | NDCG@10 |",
              "|---|---|---:|---:|---:|---:|"]
    for row in summary["results"]:
        low = row.get("low_interaction")
        if low:
            lines.append(f"| {row['domain']} | {row['model']} | {low['n_seeds']} | {low['n_users']} / {low['n_targets']} | {format_value(low,'recall@10')} | {format_value(low,'ndcg@10')} |")
    lines += ["", "## Interpretation and limits" if english else "## 解读边界", ""]
    if english:
        lines += ["Three training seeds do not establish statistical significance or a population confidence interval. A mean increase can coexist with losses in individual seeds. Report decreases and near-zero low-interaction scores without promising online gains.", "",
                  "Legacy MiniLM baselines used a fixed five-epoch main ranker budget and one shared adaptation seed. Qwen uses a separately declared warmup of up to20 epochs plus sampled adaptation and full-pipeline seeds. Differences against legacy tables mix model, budget and sampling changes; within-Qwen contrasts are the controlled objective comparison.", "",
                  "All source baseline reports and artifacts remain separate. Hash declarations are audited here; baseline byte preservation is claimed only when the separate verification evidence is supplied. Complete findings require all27 real full-evaluation records and all training evidence, rather than a visually complete table."]
    else:
        lines += ["三个训练种子不能证明统计显著性，也不能代表用户总体置信区间。均值增加可能伴随部分 seed 下降；应如实报告下降和低交互组接近零的情况，不承诺线上收益。", "",
                  "旧 MiniLM 主表采用固定5个 epoch 的推荐器预算和一个共享适配 seed；Qwen 采用另行声明的最多20 epoch warmup、抽样适配和完整流水线种子。与旧主表的差异同时包含模型、预算和采样变化，不能全部归因为模型或 LoRA；Qwen 三目标之间才是本轮受控对照。", "",
                  "旧报告与工件保持独立。本报告核对声明的哈希与训练证据；只有提供独立 baseline verification 时才声称旧工件字节未变。完成结论需要27条真实完整评价及全部证据，不能由表格外观代替。"]
    lines += ["", "## Training evidence" if english else "## 训练与导出证据", "",
              "Gradient routing is checked through the recorded informative probes (at most10), not a measurement of every step." if english else
              "梯度路径依据记录的最多10个 informative probe 检查，不能描述为每个 step 都进行了测量。", "",
              "Export numerical checks cover only the first min(4,N) live-Qwen candidates and one validation history. They do not establish full-catalog live-Qwen re-encoding equality. A separate CPU cached-feature-to-head versus serving check, when supplied, is a different test and does not rerun Qwen." if english else
              "导出数值抽查只覆盖前 min(4,N) 个实时 Qwen 候选与一条验证历史，不能解释为全目录逐项重编码等价。若另提供 CPU 上缓存原始特征经推荐头与 serving 的比较，那是独立检查，未重新运行 Qwen，须与这项抽查区分。", "",
              "| Unit | Selected epoch | Sample / eligible | Head grad max | Total LoRA grad max | BPR→LoRA max | Candidate cache error |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for unit, evidence in sorted(verification["unit_evidence"].items()):
        sampling = evidence.get("sample_metadata") or {}
        gradients = evidence["gradient_evidence"]
        values = [gradients.get(name) for name in ("head_gradient_max_norm", "total_lora_gradient_max_norm", "bpr_to_lora_max_norm")]
        cells = [f"{value:.3g}" if finite(value) else "—" for value in values]
        parity = evidence["live_cache_parity"].get("candidate_max_abs_error")
        cells.append(f"{parity:.3g}" if finite(parity) else "—")
        lines.append(f"| {unit} | {evidence['selected_epoch']} | {sampling.get('sampled_adaptation_cases','—')} / {sampling.get('eligible_review_cases','—')} | " + " | ".join(cells) + " |")
    recovery = verification.get("execution_recovery")
    if recovery:
        lines += ["", "## Actual execution recovery" if english else "## 实际运行恢复记录", "",
                  f"{recovery.get('failed_job','—')} → {recovery.get('resumed_job','—')}; {recovery.get('affected_unit','—')}.", ""]
        if recovery.get("unsaved_test_evaluation_recomputed") is True:
            lines += [("The initial test computation finished, but saving its result failed because torch.device was not JSON serializable. After recording the device with str(device), the unsaved evaluation was recomputed from the same selected training checkpoint. This is one persisted formal record for that selected checkpoint, not a claim that test computation never repeated. Existing persisted results are reused after subsequent interruptions." if english else
                       "首次测试计算已经结束，但保存结果时因 torch.device 无法 JSON 序列化而失败，未留下正式结果。将记录字段修为 str(device) 后，从同一个已选训练 checkpoint 重新计算了这次未保存的评估。这里的口径是一条已持久化的正式记录，不代表测试计算从未重复；已有保存结果在之后的恢复中直接复用。"), ""]
        if english:
            lines += [f"For the first serialization recovery, the note records unchanged training code: {recovery.get('training_code_unchanged_between_jobs')}; reused selected training checkpoints: {recovery.get('selected_training_checkpoints_reused')}; test scores used for selection or tuning: {recovery.get('test_scores_used_for_selection_or_tuning')}. The recorded changed module is `{recovery.get('only_changed_model_module_between_jobs','unknown')}`. This operational recovery does not relax any of the27-record or training-evidence audit requirements."]
        else:
            lines += [f"首次序列化恢复记录声明：训练代码未变={recovery.get('training_code_unchanged_between_jobs')}，复用已选训练 checkpoint={recovery.get('selected_training_checkpoints_reused')}，使用测试分数选模或调参={recovery.get('test_scores_used_for_selection_or_tuning')}；记录的变更模块为 `{recovery.get('only_changed_model_module_between_jobs','unknown')}`。这项运行恢复不放宽27条结果或任何训练证据的审计标准。"]
        numerical = recovery.get("numerical_recovery")
        if isinstance(numerical, dict):
            lines += ["", "### Numerical recovery and backend limits" if english else "### 数值恢复与 backend 限制", "",
                      f"{numerical.get('failed_job','—')} → diagnostic {numerical.get('diagnostic_job','—')} → {numerical.get('resumed_job','—')}; {numerical.get('affected_unit','—')}.", ""]
            if english:
                lines += [f"The recorded automatic-SDPA diagnostic first failed at step {numerical.get('diagnostic_first_nonfinite_step','unknown')}, after {numerical.get('diagnostic_successful_updates_before_failure','unknown')} successful updates: forward values and head gradients were finite, but LoRA gradients were nonfinite before clipping. The same clean state, batch and RNG reproduced the failure; MATH replay and a fresh100-step MATH diagnostic prefix remained finite. This evidence does not establish a specific fused-kernel root cause.", "",
                          f"The recovery changes live CUDA candidate training to MATH SDPA and rejects nonfinite head/LoRA gradient norms before optimizer.step; cached encoding remains automatic. The {numerical.get('completed_formal_units_preserved','unknown')} completed formal units are preserved. The failed unit restarts from initialization, without a resume checkpoint. Unchanged objective, sampling, learning rates and budget: {numerical.get('objective_sampling_learning_rates_and_budget_unchanged')}; test scores used for selection or tuning: {numerical.get('test_scores_used_for_selection_or_tuning')}.", "",
                          "All six Clothing LoRA units use MATH training SDPA; previously completed Baby/Sports LoRA units retain their automatic-SDPA execution and are not required to acquire new metadata retroactively. Clothing frozen42 is reused and has no LoRA training. These backend trajectories are not asserted bitwise identical; cross-domain differences are conditional on the numerical configurations.", "",
                          f"Numerical recovery status: {numerical.get('status','unknown')}; production same-state regression: {numerical.get('production_same_state_regression','unknown')}; production100-step prefix regression: {numerical.get('production_prefix100_regression','unknown')}. Final completeness additionally requires status=complete, both production gates=passed, and all six Clothing LoRA training.json records declaring MATH, the nonfinite gradient guard and automatic cached encoding."]
            else:
                lines += [f"记录的 automatic SDPA 诊断在第 {numerical.get('diagnostic_first_nonfinite_step','未知')} step 首次出现故障，此前完成 {numerical.get('diagnostic_successful_updates_before_failure','未知')} 次更新：前向值和推荐头梯度有限，但裁剪前 LoRA 梯度出现非有限值。同一 clean state、batch 与 RNG 可复现；MATH 重放及全新100步 MATH 诊断前缀均有限。该证据不能确定具体 fused kernel 是根因。", "",
                          f"恢复将实时 CUDA 候选训练改用 MATH SDPA，并在 optimizer.step 前拒绝非有限的推荐头/LoRA 梯度范数；缓存编码仍为 automatic。保留已经完成的 {numerical.get('completed_formal_units_preserved','未知')} 条正式记录；失败单元没有恢复 checkpoint，从初始化重启。训练目标、采样、学习率和预算未变={numerical.get('objective_sampling_learning_rates_and_budget_unchanged')}，使用测试分数选模或调参={numerical.get('test_scores_used_for_selection_or_tuning')}。", "",
                          "Clothing 六个 LoRA 单元统一使用 MATH 训练 SDPA；Baby/Sports 已完成 LoRA 单元保留 automatic SDPA 原运行，不要求事后补写新元数据。Clothing frozen42 复用且没有 LoRA 训练。这些 backend 轨迹不声明 bitwise 等价，跨领域差异的解释受不同数值配置限制。", "",
                          f"数值恢复状态：{numerical.get('status','未知')}；生产函数同状态回归：{numerical.get('production_same_state_regression','未知')}；生产函数100步前缀回归：{numerical.get('production_prefix100_regression','未知')}。最终完成还要求状态为 complete、两个生产门槛均为 passed，并核对 Clothing 六个 LoRA 的 training.json 声明 MATH、非有限梯度保护与 automatic 缓存编码。"]
        lines += ["", "Source: `experiments/qwen/recovery_note.json` (included verbatim in the machine-readable verification)."]
    lines += ["", "## Audit" if english else "## 审计状态", ""]
    for issue in verification["errors"][:40]:
        lines.append(f"- {issue['unit']} / {issue['check']}: {issue['message']}")
    for warning in verification["warnings"]:
        lines.append(f"- {warning}")
    if verification["missing_units"]:
        lines.append("- Missing units: " + ", ".join(f"{domain}/{mode}/seed{seed}" for domain, mode, seed in verification["missing_units"]))
    lines += ["", "## Sources" if english else "## 资料", "",
              f"- [Pinned Qwen model](https://huggingface.co/{MODEL_ID}/tree/{REVISION})",
              "- [Amazon2014 original data](https://mcauleylab.ucsd.edu/public_datasets/data/amazon/datasets.html)",
              "- [CLIP original paper](https://proceedings.mlr.press/v139/radford21a.html)",
              "- Machine-readable run evidence: `experiments/qwen/verification.json` and `artifacts/qwen/<domain>/<mode>/seed<seed>/training.json`."]
    return "\n".join(lines) + "\n"


def build_report(root: Path, *, verify_cache_files: bool = False) -> dict:
    rows, verification = audit_root(Path(root), verify_cache_files=verify_cache_files)
    summaries, paired = aggregate(rows)
    return {"schema_version": 1, "complete": verification["complete"], "raw_record_count": len(rows),
            "results": summaries, "paired_comparisons": paired, "verification": verification}


def atomic_write(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def write_reports(root: Path, summary: dict) -> None:
    root = Path(root)
    output = root / "deliverables"
    atomic_write(output / "ShopLens_Qwen_report.md", render_report(summary))
    atomic_write(output / "ShopLens_Qwen_report_en.md", render_report(summary, english=True))
    atomic_write(output / "results_qwen_summary.json", json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    buffer = io.StringIO()
    fields = ["domain", "model", "n_seeds", *(key for metric in METRICS for key in (metric, metric + "_std"))]
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(summary["results"])
    atomic_write(output / "results_qwen_summary.csv", buffer.getvalue())
    atomic_write(root / "experiments/qwen/verification.json", json.dumps(summary["verification"], indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--verify-cache-files", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    summary = build_report(args.root, verify_cache_files=args.verify_cache_files)
    if not args.check_only:
        write_reports(args.root, summary)
    print(json.dumps({"complete": summary["complete"], "raw_record_count": summary["raw_record_count"],
                      "groups": len(summary["results"]), "audit_errors": len(summary["verification"]["errors"])}))


if __name__ == "__main__":
    main()
